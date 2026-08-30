// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() { }

function getGallerySelectedIndex() { }

function request(url, data, handler, errorHandler) {
    let xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    let js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    let js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? "0" + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return (
            pad2(Math.floor(secs / 60 / 60)) +
            ":" +
            pad2(Math.floor(secs / 60) % 60) +
            ":" +
            pad2(Math.floor(secs) % 60)
        );
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}

let originalAppTitle = undefined;

onUiLoaded(function () {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    let title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = "[" + progress.trim() + "] " + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}

function randomId() {
    return (
        "task(" +
        Math.random().toString(36).slice(2, 7) +
        Math.random().toString(36).slice(2, 7) +
        Math.random().toString(36).slice(2, 7) +
        ")"
    );
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(
    id_task,
    progressbarContainer,
    gallery,
    atEnd,
    onProgress,
    inactivityTimeout = 40,
) {
    let dateStart = new Date();
    let wasEverActive = false;
    let parentProgressbar = progressbarContainer.parentNode;
    let wakeLock = null;

    if (gallery && gallery.classList.contains("hidden"))
        gallery = gallery.parentElement.querySelector(".gradio-video");

    let requestWakeLock = async function () {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock) return;
        try {
            wakeLock = await navigator.wakeLock.request("screen");
        } catch (err) {
            console.error("Wake Lock is not supported.");
        }
    };

    let releaseWakeLock = async function () {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error("Wake Lock release failed", err);
        }
    };

    let divProgress = document.createElement("div");
    divProgress.className = "progressDiv";
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    let divInner = document.createElement("div");
    divInner.className = "progress";

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    let livePreview = null;
    let idLivePreview = 0;
    let consecutiveErrors = 0;

    let removeProgressBar = function () {
        releaseWakeLock();
        if (!divProgress) return;

        setTitle("");
        divProgress.remove();
        if (livePreview) livePreview.remove();
        atEnd();

        divProgress = null;
    };

    let updateLivePreview = function (res) {
        if (!res.live_preview || !gallery) return;

        const img = new Image();
        img.onload = function () {
            // Image decoding is asynchronous; generation may finish first.
            if (!divProgress || !gallery) return;

            if (!livePreview) {
                livePreview = document.createElement("div");
                livePreview.className = "livePreview";
                gallery.insertBefore(livePreview, gallery.firstElementChild);
            }

            livePreview.appendChild(img);
            while (livePreview.childElementCount > 2) {
                livePreview.firstElementChild.remove();
            }
        };
        img.src = res.live_preview;
    };

    let funProgress = function () {
        if (!divProgress) return;

        requestWakeLock();
        request(
            "./internal/progress",
            {
                id_task: id_task,
                id_live_preview: idLivePreview,
                live_preview: Boolean(gallery),
            },
            function (res) {
                if (!divProgress) return;
                consecutiveErrors = 0;

                if (res.completed) {
                    removeProgressBar();
                    return;
                }

                if (res.id_live_preview != null) {
                    idLivePreview = res.id_live_preview;
                }
                updateLivePreview(res);

                let progressText = "";

                divInner.style.width = (res.progress || 0) * 100.0 + "%";
                divInner.style.background = res.progress ? "" : "transparent";

                if (res.progress > 0) {
                    progressText = ((res.progress || 0) * 100.0).toFixed(0) + "%";
                }

                setTitle(progressText);

                // [NEO] "12/30 · 3.4 s/it · ETA 1:02" is a lot more informative than a
                // bare percentage, and it is the information people want when tuning
                // samplers, resolutions and model settings.
                let details = [];

                if (res.total_steps > 0) {
                    let stepText = (res.steps || 0) + "/" + res.total_steps;
                    if (res.job_count > 1) {
                        stepText = (res.job_no || 0) + 1 + "/" + res.job_count + " · " + stepText;
                    }
                    details.push(stepText);
                }

                if (res.rate && res.rate > 0.01) {
                    if (res.rate >= 1) {
                        details.push(res.rate.toFixed(1) + " it/s");
                    } else {
                        details.push((1 / res.rate).toFixed(1) + " s/it");
                    }
                }

                if (res.eta) {
                    details.push("ETA " + formatTime(res.eta));
                }

                let suffix = details.length ? " · " + details.join(" · ") : "";

                if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
                    progressText = res.textinfo + " " + progressText + suffix;
                } else {
                    progressText = progressText + suffix;
                }

                divInner.textContent = progressText;

                const elapsedFromStart = (Date.now() - dateStart.getTime()) / 1000;

                if (res.active) wasEverActive = true;

                if (!res.active && wasEverActive) {
                    removeProgressBar();
                    return;
                }

                if (
                    elapsedFromStart > inactivityTimeout &&
                    !res.queued &&
                    !res.active
                ) {
                    removeProgressBar();
                    return;
                }

                if (onProgress) {
                    onProgress(res);
                }

                setTimeout(funProgress, opts.live_preview_refresh_period || 500);
            },
            function () {
                // Brief network hiccups should not make a running job appear done.
                consecutiveErrors += 1;
                if (consecutiveErrors <= 3 && divProgress) {
                    setTimeout(funProgress, 500);
                } else {
                    removeProgressBar();
                }
            },
        );
    };

    funProgress();
}
