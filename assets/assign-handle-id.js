document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".rc-slider-handle").forEach((h, i) => {
    h.id = "handle-" + i;
  });
});
