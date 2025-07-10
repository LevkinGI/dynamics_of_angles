document.addEventListener("DOMContentLoaded", () => {
  // 1. Обходим все слайдеры Dash (div с data-dash-is-loading, id="alpha-scale-slider", …)
  document.querySelectorAll(".rc-slider").forEach(slider => {
    const handle = slider.querySelector(".rc-slider-handle");
    if (!handle) return;

    // 2. Берём id родительского div'а (он совпадает с id, который задали в dcc.Slider)
    const parentId = slider.getAttribute("id");
    if (!parentId) return;                 // на всякий случай

    // 3. Делаем из него «ручечный» id
    handle.id = `${parentId}-handle`;      // alpha-scale-slider-handle, H-slider-handle, …
  });
});
