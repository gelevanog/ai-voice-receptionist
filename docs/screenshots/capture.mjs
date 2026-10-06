// Screenshots of the dashboard with headless Chrome (puppeteer-core).
//
//   node docs/screenshots/capture.mjs <base-url> <caller.wav> [pages...]
//
// "live" places a real browser call: Chrome's fake microphone plays <caller.wav> (a simulated caller, see
// README > Screenshots) into the page, which streams it to the server like a real microphone would.
// Other pages: calls, call:<id>, calendar:<week>, eval.
import { createRequire } from "module";

const require = createRequire(process.env.PUPPETEER_FROM || import.meta.url);
const puppeteer = require("puppeteer-core");
const [base, wav, ...pages] = process.argv.slice(2);
const out = new URL(".", import.meta.url).pathname;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || "/usr/bin/google-chrome",
  headless: "new",
  args: [
    "--no-sandbox",
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}%noloop`,
    "--autoplay-policy=no-user-gesture-required",
  ],
});

async function shot(path, name, { height = 1000, wait = 0, full = false } = {}) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height, deviceScaleFactor: 1.5 });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
  await page.goto(base + path, { waitUntil: "networkidle0" });
  if (wait) await sleep(wait);
  await page.screenshot({ path: `${out}${name}.png`, fullPage: full });
  console.log(`${name}.png`);
  return page;
}

for (const item of pages) {
  if (item.startsWith("live")) {
    const seconds = Number(item.split(":")[1] || 100);
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1.5 });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
    page.on("console", (msg) => { if (msg.type() === "error") console.log("page error:", msg.text()); });
    await page.goto(base + "/", { waitUntil: "networkidle0" });
    await page.click("#start");
    for (let t = 0; t < seconds; t += 5) {
      await sleep(5000);
      const ended = await page.$eval("#transcript", (el) => el.textContent.includes("call ended"));
      if (ended) break;
    }
    await sleep(1500);
    // Compact hero: the transcript panel scrolled to the booking part, the waterfall beside it.
    await page.setViewport({ width: 1440, height: 1180, deviceScaleFactor: 1.5 });
    await page.evaluate(() => {
      const t = document.querySelector("#transcript");
      t.style.maxHeight = "820px";
      const tools = [...t.querySelectorAll(".tool")];
      const anchor = tools.length > 1 ? tools[1] : tools[0];
      if (anchor) t.scrollTop = anchor.offsetTop - t.offsetTop - 120;
    });
    await page.screenshot({ path: `${out}live-call.png` });
    await page.evaluate(() => { document.querySelector("#transcript").style.maxHeight = "none"; });
    await page.screenshot({ path: `${out}live-call-full.png`, fullPage: true });
    console.log("live-call.png, live-call-full.png");
  } else if (item === "calls") {
    await shot("/calls", "call-history", { full: true });
  } else if (item.startsWith("call:")) {
    await shot(`/calls/${item.slice(5)}`, "call-detail", { full: true });
  } else if (item.startsWith("calendar")) {
    const week = item.split(":")[1];
    await shot(`/calendar${week ? `?week=${week}` : ""}`, "calendar", { full: true });
  } else if (item === "eval") {
    await shot("/eval", "evaluation", { full: true });
  }
}
await browser.close();
