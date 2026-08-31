// Small helpers for the Civitai browser tab.
// Loaded from extensions-builtin/sd_forge_civitai/javascript/ on startup.

function civitaiCardInput() {
    var box = gradioApp().getElementById("civitai_selected_model");
    if (!box) return null;
    return box.querySelector("textarea") || box.querySelector("input");
}

function civitaiPickCard(card) {
    var input = civitaiCardInput();
    if (!input) return;

    gradioApp()
        .querySelectorAll(".civitai-card")
        .forEach(function (c) {
            c.classList.remove("civitai-card-selected");
        });

    card.classList.add("civitai-card-selected");
    input.value = card.getAttribute("data-civitai-id");
    updateInput(input);

    // A hidden button click is the most reliable way to get Gradio to send the
    // (just updated) value back to Python.
    var btn = gradioApp().getElementById("civitai_select_btn");
    if (btn) btn.click();
}

function civitaiCurrentTab() {
    var t2i = gradioApp().getElementById("tab_txt2img");
    if (t2i && t2i.style.display !== "none") return "txt2img";

    var i2i = gradioApp().getElementById("tab_img2img");
    if (i2i && i2i.style.display !== "none") return "img2img";

    return "txt2img";
}

function civitaiCopyWords(link) {
    var words = link.getAttribute("data-words");
    if (!words) return;

    var tabname = civitaiCurrentTab();
    var textarea = gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");
    if (!textarea) return;

    if (typeof updatePromptArea === "function") {
        updatePromptArea(words, textarea);
    } else {
        textarea.value = textarea.value + ", " + words;
        updateInput(textarea);
    }

    link.textContent = "copied!";
    setTimeout(function () {
        link.textContent = "copy trigger words to the positive prompt";
    }, 1500);
}

// Update checking: "Check for updates" starts a background thread in Python,
// this poll keeps the tab in sync until the thread reports it is done.

function civitaiUpdatesRunning() {
    var box = gradioApp().getElementById("civitai_updates_status");
    var marker = box && box.querySelector("[data-running]");
    return marker ? marker.getAttribute("data-running") === "1" : false;
}

function civitaiPollUpdates() {
    if (!civitaiUpdatesRunning()) {
        return;
    }

    var btn = gradioApp().getElementById("civitai_updates_poll");
    if (btn) {
        btn.click();
    }

    setTimeout(civitaiPollUpdates, 1200);
}

function civitaiUpdate(button) {
    var box = gradioApp().getElementById("civitai_update_pick");
    if (!box) return;

    var input = box.querySelector("textarea") || box.querySelector("input");
    if (!input) return;

    input.value = button.getAttribute("data-hash");
    updateInput(input);

    var btn = gradioApp().getElementById("civitai_update_pick_btn");
    if (btn) btn.click();

    button.disabled = true;
    button.textContent = "queued";
}

onUiUpdate(function () {
    var start = gradioApp().getElementById("civitai_update_check");
    if (!start || start.dataset.pollHooked) return;

    start.dataset.pollHooked = "1";
    start.addEventListener("click", function () {
        setTimeout(civitaiPollUpdates, 1000);
    });
});
