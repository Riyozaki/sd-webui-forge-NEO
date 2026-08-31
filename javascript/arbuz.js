// ArbuzDiffusion: turn the flags the Python side baked into the page into
// classes on the Gradio root, so the stylesheet can follow the theme weight
// and the interface density.

function arbuzApplyFlags() {
    var root = gradioApp();
    if (!root) return;

    var flags = root.querySelector("#arbuz-flags");
    if (!flags || flags.dataset.applied === "1") return;

    flags.dataset.applied = "1";

    var classes = root.classList;
    classes.remove("arbuz-density-compact", "arbuz-density-cozy", "arbuz-density-spacious");
    classes.remove("arbuz-weight-dark", "arbuz-weight-light");

    classes.add("arbuz-density-" + (flags.getAttribute("data-density") || "cozy"));
    classes.add("arbuz-weight-" + (flags.getAttribute("data-weight") || "dark"));
}

onUiUpdate(arbuzApplyFlags);
