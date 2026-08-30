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
