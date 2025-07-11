// // assign-handle-id.js
// // Назначаем каждому бегунку (ручке) id вида "<id-слайдера>-handle",
// // чтобы на него можно было повесить dbc.Tooltip.

// document.addEventListener("DOMContentLoaded", () => {
//   // 1. Берём ВСЕ элементы-бегунки, которые создаёт rc-slider.
//   document.querySelectorAll(".rc-slider-handle").forEach(handle => {

//     // 2. Поднимаемся вверх по дереву и ищем ближайший <div>, у которого есть id —
//     //    это тот самый контейнер, который Dash сгенерировал на основе
//     //    dcc.Slider(id="...").
//     const sliderDiv = handle.closest("div[id]");   // <div id="alpha-scale-slider">…<div class="rc-slider">…
//     if (!sliderDiv) return;                        // на случай экзотической вёрстки

//     // 3. Формируем новый id для ручки и присваиваем.
//     handle.id = `${sliderDiv.id}-handle`;          // получаем alpha-scale-slider-handle
//   });
// });
