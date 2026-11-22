const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#image");
const dropzone = document.querySelector("#dropzone");
const uploadContent = document.querySelector("#upload-content");
const previewContent = document.querySelector("#preview-content");
const previewImage = document.querySelector("#image-preview");
const previewName = document.querySelector("#preview-name");
const fileError = document.querySelector("#file-error");
const submitButton = document.querySelector("#submit-button");
const maxFileSize = 8 * 1024 * 1024;
const allowedExtensions = [".jpg", ".jpeg", ".png", ".webp"];
let previewUrl;

function showError(message) {
    fileError.textContent = message;
    fileError.hidden = false;
}

function showPreview(file) {
    fileError.hidden = true;
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();

    if (!allowedExtensions.includes(extension)) {
        showError("Choose a JPG, PNG, or WebP image.");
        fileInput.value = "";
        return;
    }
    if (file.size > maxFileSize) {
        showError("That image is larger than 8 MB. Choose a smaller file.");
        fileInput.value = "";
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    previewName.textContent = file.name;
    uploadContent.hidden = true;
    previewContent.hidden = false;
}

fileInput.addEventListener("change", () => {
    const [file] = fileInput.files;
    if (file) {
        showPreview(file);
    }
});

for (const eventName of ["dragenter", "dragover"]) {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
    });
}

for (const eventName of ["dragleave", "drop"]) {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
    });
}

dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) {
        return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    showPreview(file);
});

form.addEventListener("submit", (event) => {
    const [file] = fileInput.files;
    if (!file) {
        event.preventDefault();
        showError("Choose an image before starting.");
        return;
    }
    if (file.size > maxFileSize) {
        event.preventDefault();
        showError("That image is larger than 8 MB. Choose a smaller file.");
        return;
    }

    submitButton.disabled = true;
    submitButton.querySelector("span").textContent = "Creating your segmentation…";
});
