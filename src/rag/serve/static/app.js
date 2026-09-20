/* The page's whole behaviour: read the service status and the build facts
   once, submit a query, draw whatever came back, and start a query named in
   the URL.

   The answer is written into the page as text and never as markup, so a model
   that emits HTML-ish prose cannot put anything into the document. */

const byId = (id) => document.getElementById(id);

const el = {
  form: byId("form"),
  prompt: byId("prompt"),
  submit: byId("submit"),
  fieldError: byId("field-error"),
  chip: byId("chip"),
  chipText: byId("chip-text"),
  about: byId("about"),
  aboutBody: byId("about-body"),
  result: byId("result"),
  echo: byId("echo"),
  recipePanel: byId("recipe-panel"),
  recipeTitle: byId("recipe-title"),
  recipeMeta: byId("recipe-meta"),
  provenance: byId("provenance"),
  ingredientsBlock: byId("ingredients-block"),
  ingredients: byId("ingredients"),
  stepsBlock: byId("steps-block"),
  steps: byId("steps"),
  notesBlock: byId("notes-block"),
  notes: byId("notes"),
  remainderNotice: byId("remainder-notice"),
  remainder: byId("remainder"),
  answerPanel: byId("answer-panel"),
  answer: byId("answer"),
  actions: byId("result-actions"),
  another: byId("another"),
  copy: byId("copy"),
  failurePanel: byId("failure-panel"),
  failureTitle: byId("failure-title"),
  failureBody: byId("failure-body"),
  failureCode: byId("failure-code"),
  retry: byId("retry"),
};

let lastQuery = "";
let inFlight = false;
/* What "Copy recipe" copies, which is the clean text of whatever the page is
   showing: the laid-out recipe, or the model's own words when nothing parsed. */
let copyText = "";
/* /ui/info, read once. Both the provenance line and the about panel are
   built from it, and neither is shown until it arrives. */
let buildInfo = null;

/* Copy for the failures /ui/query can come back with, plus the two the page
   raises itself: an unreachable service, and a 200 carrying no answer, which
   no envelope code covers. */
const FAILURE_COPY = {
  no_snapshot: {
    title: "The kitchen is being restocked",
    body: "The service has no recipe snapshot loaded right now, so it cannot write anything. This is on our side, not your request.",
  },
  internal_error: {
    title: "That one did not work",
    body: "Something broke while the recipe was being written. Nothing was saved, so trying again starts clean.",
  },
  empty_answer: {
    title: "No recipe came back",
    body: "The model returned nothing this time. That is usually a one-off, so try the same request again.",
  },
  network_error: {
    title: "The service could not be reached",
    body: "The request never got an answer. Check that the service is still running, then try again.",
  },
};

const AUTH_COPY = {
  title: "The page cannot reach the kitchen",
  body: "The service refused the request, so nothing was written. This one is a deployment problem rather than anything about your request.",
};

const FALLBACK_COPY = {
  title: "That one did not work",
  body: "Something went wrong while the recipe was being written. Trying again starts clean.",
};

function failureCopy(code) {
  if (code === "unauthenticated" || code === "forbidden") return AUTH_COPY;
  return FAILURE_COPY[code] || FALLBACK_COPY;
}

function codeFrom(body, fallback) {
  if (body && body.error && typeof body.error.code === "string") {
    return body.error.code;
  }
  return fallback;
}

/* ── the status chip ─────────────────────────────────────────────────── */

function setChip(state, text) {
  el.chip.dataset.state = state;
  el.chipText.textContent = text;
}

async function checkService() {
  try {
    const response = await fetch("/ui/health");
    const body = await response.json().catch(() => null);
    const healthy = response.ok && body && body.status === "ok";
    setChip(healthy ? "ok" : "down", healthy ? "service ok" : "service not ready");
  } catch (error) {
    setChip("down", "service unreachable");
  }
}

/* ── the build the page is talking to ────────────────────────────────── */

function aboutRow(term, value) {
  const row = document.createElement("div");
  row.className = "about-row";
  const label = document.createElement("span");
  label.className = "about-term";
  label.textContent = term;
  const text = document.createElement("span");
  text.className = "about-value";
  text.textContent = value === null || value === undefined ? "" : String(value);
  row.append(label, text);
  return row;
}

function aboutGroup(heading) {
  const group = document.createElement("p");
  group.className = "about-group";
  group.textContent = heading;
  return group;
}

/* The provenance line and the about panel, both from /info. Neither is drawn
   until the facts arrive: a line of punctuation under a recipe would be worse
   than no line at all. */
function applyBuildInfo() {
  if (!buildInfo || !buildInfo.snapshot) return;
  const snapshot = buildInfo.snapshot;
  const models = buildInfo.models || {};
  const chunks = buildInfo.chunk_sets || {};

  el.provenance.textContent = [
    "Snapshot " + snapshot.created_at,
    snapshot.retrieval_strategy + " retrieval",
    snapshot.dataset_key,
    models.generator,
  ]
    .filter(Boolean)
    .join(" · ");
  el.provenance.hidden = false;

  el.aboutBody.replaceChildren(
    aboutGroup("snapshot"),
    aboutRow("created", snapshot.created_at),
    aboutRow("retrieval", snapshot.retrieval_strategy),
    aboutRow("dataset", snapshot.dataset_key),
    aboutGroup("models"),
    aboutRow("encoder", models.encoder),
    aboutRow("generator", models.generator),
    aboutGroup("chunk sets"),
    aboutRow("dense", (chunks.dense || []).join(", ")),
    aboutRow("bm25", (chunks.bm25 || []).join(", ")),
  );
  el.about.hidden = false;
}

async function loadBuildInfo() {
  try {
    const response = await fetch("/ui/info");
    if (!response.ok) return;
    buildInfo = await response.json();
    applyBuildInfo();
  } catch (error) {
    /* The provenance line and the about panel are the only things fed from
       here, so a failure leaves the page whole and just quieter. */
  }
}

/* ── rendering ───────────────────────────────────────────────────────── */

function badge(text) {
  const pill = document.createElement("li");
  pill.className = "badge badge-sm";
  pill.textContent = text;
  return pill;
}

/* Says nothing about the dish, only that this reply just arrived, so it is
   set quieter than the facts beside it rather than as one of them. */
function freshNote() {
  const note = document.createElement("li");
  note.className = "fresh";
  note.textContent = "generated just now";
  return note;
}

function ingredientEntry(entry) {
  const row = document.createElement("li");
  if (entry.kind === "group") {
    row.className = "ingredient-group";
    row.textContent = entry.heading;
    return row;
  }
  row.className = "ingredient";
  const amount = document.createElement("span");
  amount.className = "ingredient-qty";
  amount.textContent = entry.quantity || "";
  const name = document.createElement("span");
  name.textContent = entry.text;
  row.append(amount, name);
  return row;
}

function stepEntry(entry) {
  const row = document.createElement("li");
  if (entry.kind === "group") {
    row.className = "method-group";
    row.textContent = entry.heading;
  } else {
    row.className = "method-step";
    row.textContent = entry.text;
  }
  return row;
}

function noteEntry(note) {
  const row = document.createElement("li");
  row.className = "note";
  row.textContent = note;
  return row;
}

/* The answer as the page lays it out. Every string goes in as text and never
   as markup, so a model that writes HTML-ish prose cannot put anything into
   the document. */
function renderRecipe(recipe) {
  el.recipeTitle.textContent = recipe.title || "";
  el.recipeTitle.hidden = !recipe.title;

  el.recipeMeta.replaceChildren(
    ...recipe.meta.map((fact) => badge(fact.label + ": " + fact.value)),
    freshNote(),
  );

  el.ingredients.replaceChildren(...recipe.ingredients.map(ingredientEntry));
  el.ingredientsBlock.hidden = recipe.ingredients.length === 0;

  el.steps.replaceChildren(...recipe.steps.map(stepEntry));
  el.stepsBlock.hidden = recipe.steps.length === 0;

  el.notes.replaceChildren(...recipe.notes.map(noteEntry));
  el.notesBlock.hidden = recipe.notes.length === 0;

  el.remainder.textContent = recipe.remainder || "";
  el.remainder.hidden = !recipe.remainder;
  el.remainderNotice.hidden = !recipe.remainder;

  el.recipePanel.hidden = false;
  el.answerPanel.hidden = true;
  el.failurePanel.hidden = true;
  el.actions.hidden = false;
  applyBuildInfo();
}

/* The same recipe as text, for the clipboard. Built from the laid-out shape
   rather than from the raw answer, so what is pasted is what was on screen. */
function recipeToText(recipe) {
  const blocks = [];

  const facts = recipe.meta.map((fact) => fact.label + ": " + fact.value);
  if (recipe.title) facts.unshift(recipe.title);
  if (facts.length) blocks.push(facts.join("\n"));

  const ingredients = recipe.ingredients.map((entry) =>
    entry.kind === "group"
      ? entry.heading
      : "- " + [entry.quantity, entry.text].filter(Boolean).join(" "),
  );
  if (ingredients.length) {
    blocks.push(["Ingredients", ...ingredients].join("\n"));
  }

  let number = 0;
  const steps = recipe.steps.map((entry) => {
    if (entry.kind === "group") return entry.heading;
    number += 1;
    return number + ". " + entry.text;
  });
  if (steps.length) blocks.push(["Method", ...steps].join("\n"));

  if (recipe.notes.length) {
    blocks.push(["Notes", ...recipe.notes.map((note) => "- " + note)].join("\n"));
  }
  if (recipe.remainder) blocks.push(recipe.remainder);

  return blocks.join("\n\n");
}

function showAnswer(answer) {
  el.answer.textContent = answer;
  copyText = answer;
  el.recipePanel.hidden = true;
  el.answerPanel.hidden = false;
  el.failurePanel.hidden = true;
  el.actions.hidden = false;
}

function showFailure(code) {
  const copy = failureCopy(code);
  el.failureTitle.textContent = copy.title;
  el.failureBody.textContent = copy.body;
  el.failureCode.textContent = code;
  el.failurePanel.hidden = false;
  el.answerPanel.hidden = true;
  el.recipePanel.hidden = true;
  el.actions.hidden = true;
}

function setBusy(busy) {
  inFlight = busy;
  el.submit.textContent = busy ? "Writing the recipe..." : "Generate recipe";
  el.form.setAttribute("aria-busy", String(busy));
  syncSubmitState();
}

/* The button is disabled while the prompt is empty, which is what keeps most
   people from ever meeting the 422 the API answers for an empty query. */
function syncSubmitState() {
  el.submit.disabled = inFlight || el.prompt.value.trim() === "";
}

/* Move the view to the answer as it arrives. The entry screen stays where it
   is, above, so a second question is one scroll away and nothing is hidden;
   this only stops the answer landing below the fold. */
function revealResult() {
  el.result.scrollIntoView({ block: "start" });
}

/* ── the request ─────────────────────────────────────────────────────── */

async function generate(query) {
  if (inFlight) return;
  lastQuery = query;

  el.echo.textContent = "You asked for: " + query;
  el.result.hidden = false;
  el.recipePanel.hidden = true;
  el.answerPanel.hidden = true;
  el.failurePanel.hidden = true;
  el.actions.hidden = true;
  copyText = "";
  setBusy(true);

  try {
    const response = await fetch("/ui/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: query }),
    });
    const body = await response.json().catch(() => null);

    if (!response.ok) {
      showFailure(codeFrom(body, "internal_error"));
    } else if (!body || typeof body.answer !== "string" || !body.answer.trim()) {
      showFailure("empty_answer");
    } else if (body.recipe && body.recipe.parsed) {
      copyText = recipeToText(body.recipe);
      renderRecipe(body.recipe);
    } else {
      showAnswer(body.answer);
    }
  } catch (error) {
    showFailure("network_error");
  } finally {
    setBusy(false);
    revealResult();
  }
}

/* ── wiring ──────────────────────────────────────────────────────────── */

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = el.prompt.value.trim();
  if (!query) {
    el.fieldError.hidden = false;
    el.prompt.focus();
    return;
  }
  el.fieldError.hidden = true;
  generate(query);
});

el.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.form.requestSubmit();
  }
});

el.prompt.addEventListener("input", () => {
  el.fieldError.hidden = true;
  syncSubmitState();
});

for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    el.prompt.value = example.textContent.trim();
    syncSubmitState();
    el.form.requestSubmit();
  });
}

el.retry.addEventListener("click", () => {
  if (lastQuery) generate(lastQuery);
});

/* The one action with a server cost, so it reloads into a fresh request for
   the same query rather than patching the view: what comes back is another
   recipe, not a variation of this one, and the URL stays honest about it. */
el.another.addEventListener("click", () => {
  const query = lastQuery || el.prompt.value.trim();
  if (!query) return;
  window.location.assign("/?q=" + encodeURIComponent(query));
});

el.copy.addEventListener("click", async () => {
  const label = el.copy.textContent;
  try {
    await navigator.clipboard.writeText(copyText);
    el.copy.textContent = "Copied";
  } catch (error) {
    el.copy.textContent = "Copy failed";
  }
  setTimeout(() => {
    el.copy.textContent = label;
  }, 1500);
});

/* A query in the URL runs on load, which is how "generate another" reloads
   into a fresh request rather than patching the view in place. */
const requested = new URLSearchParams(window.location.search).get("q");
if (requested && requested.trim()) {
  el.prompt.value = requested.trim();
  syncSubmitState();
  el.form.requestSubmit();
}

syncSubmitState();
checkService();
loadBuildInfo();
