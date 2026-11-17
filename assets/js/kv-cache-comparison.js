(() => {
  const viewer = document.getElementById("kv-comparison");
  if (!viewer) return;
  const cases = JSON.parse(document.getElementById("kv-case-data").textContent);
  const root = document.getElementById("kv-video-root").dataset.root;
  const pagination = document.getElementById("kv-pagination");
  const images = [document.getElementById("kv-left"), document.getElementById("kv-right")];
  const downloads = [document.getElementById("kv-left-mp4"), document.getElementById("kv-right-mp4")];
  const previous = document.getElementById("kv-previous");
  const next = document.getElementById("kv-next");
  const play = document.getElementById("kv-play");
  const status = document.getElementById("kv-status");
  const initializationButtons = [...viewer.querySelectorAll("[data-initialization]")];
  const trainingLabel = document.getElementById("kv-training-label");
  const initializationNames = { ode: "Causal Forcing · ODE", cd: "Causal Forcing++ · CD" };
  let initialization = "cd";
  const familyRoot = () => root + initialization + "/";
  let index = 0;
  let running = false;
  let generation = 0;
  let replay = 0;
  let timer;
  const announce = (message) => {
    status.textContent = `Example ${index + 1} of ${cases.length}. ${message}`;
  };
  const poster = () => familyRoot() + cases[index].id + "-pair-hq.jpg";
  const stop = () => {
    generation += 1;
    clearTimeout(timer);
    running = false;
    images.forEach((image) => (image.src = poster()));
    play.disabled = false;
    play.textContent = "Play both";
    play.setAttribute("aria-pressed", "false");
  };
  const loadCase = () => {
    stop();
    downloads[0].href = familyRoot() + cases[index].id + "-original.mp4";
    downloads[1].href = familyRoot() + cases[index].id + "-detached.mp4";
    trainingLabel.textContent = `${initializationNames[initialization]} → DMD · 750 iterations · EMA`;
    initializationButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.initialization === initialization)));
    images[0].alt = `${initialization.toUpperCase()} initialization, after 750 DMD iterations without KV cache detachment`;
    images[1].alt = `${initialization.toUpperCase()} initialization, after 750 DMD iterations with KV cache detachment`;
    pagination.textContent = `Example ${index + 1} / ${cases.length}`;
    previous.disabled = index === 0;
    next.disabled = index === cases.length - 1;
    document.getElementById("kv-prompt-text").textContent = cases[index].prompt;
    announce("Ready to play.");
  };
  const animate = () => {
    stop();
    const version = generation;
    const preview = new Image();
    // Both panels crop the same paired animation, sharing its frame sequence.
    // A new replay URL restarts the animation rather than reusing its timeline.
    const url = familyRoot() + cases[index].id + `-pair-hq.webp?play=${++replay}`;
    play.disabled = true;
    announce("Loading previews…");
    const fail = () => {
      if (version !== generation) return;
      stop();
      announce("The preview could not load. Please try again.");
    };
    timer = setTimeout(fail, 60000);
    preview.onerror = fail;
    preview.onload = () => {
      if (version !== generation) return;
      clearTimeout(timer);
      images.forEach((image) => (image.src = url));
      running = true;
      play.disabled = false;
      play.textContent = "Stop";
      play.setAttribute("aria-pressed", "true");
      announce("Playing both previews.");
    };
    preview.src = url;
  };
  initializationButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (initialization === button.dataset.initialization) return;
      initialization = button.dataset.initialization;
      loadCase();
    });
  });
  previous.addEventListener("click", () => {
    if (index > 0) {
      index -= 1;
      loadCase();
    }
  });
  next.addEventListener("click", () => {
    if (index < cases.length - 1) {
      index += 1;
      loadCase();
    }
  });
  play.addEventListener("click", () => {
    if (running) {
      stop();
      announce("Stopped.");
    } else animate();
  });
  document.getElementById("kv-restart").addEventListener("click", animate);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && running) {
      stop();
      announce("Playback stopped while the page was hidden.");
    }
  });
  loadCase();
})();
