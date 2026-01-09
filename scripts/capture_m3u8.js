#!/usr/bin/env node
'use strict';

// Headless helper to capture an m3u8 URL from a Kick page using Playwright.
// STDOUT contract (consumed by Rust):
//   line 1: m3u8 URL (first line containing ".m3u8")
//   optional: line starting with "COOKIES: " containing a semicolon-joined cookie header
// All diagnostic logs go to STDERR.

const { chromium } = require('playwright');

const pageUrl = process.argv[2];
if (!pageUrl) {
  console.error('usage: node scripts/capture_m3u8.js <page_url>');
  process.exit(1);
}

const timeoutMs = Number(process.env.M3U8_SCRAPE_TIMEOUT_MS || 15000);
const settleAfterFirstMs = Number(process.env.M3U8_WAIT_AFTER_FIRST_MS || 2000);
const shouldClickPlay = (process.env.M3U8_CLICK_PLAY || '1') !== '0';
const ua =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

const cookieHeader = process.env.COOKIE_HEADER || process.env.KICK_COOKIE || '';
const extraHeaders = {
  'User-Agent': ua,
};
if (cookieHeader) {
  extraHeaders['Cookie'] = cookieHeader;
}

const seen = new Set();
const candidates = [];

const normalize = (s) => s.replace(/\\u0026/g, '&').replace(/&amp;/g, '&').replace(/\\\//g, '/');
const score = (u) => {
  let scoreVal = 0;
  if (u.includes('supported_codecs')) scoreVal += 3;
  if (u.includes('player_version')) scoreVal += 2;
  if (u.includes('browser_family')) scoreVal += 1;
  scoreVal += Math.floor(u.length / 100);
  return scoreVal;
};

function consider(url) {
  const normalized = normalize(url);
  if (!normalized.toLowerCase().includes('.m3u8')) return;
  if (seen.has(normalized)) return;
  seen.add(normalized);
  candidates.push(normalized);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent: ua,
    extraHTTPHeaders: extraHeaders,
  });
  const page = await context.newPage();

  page.on('request', (req) => consider(req.url()));
  page.on('response', (resp) => consider(resp.url()));

  try {
    await page.goto(pageUrl, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    if (shouldClickPlay) {
      const playSelectors = [
        "button[aria-label='Play']",
        "button[data-testid='player-overlay-button']",
        "button:has-text('Watch')",
        "button:has-text('Play')",
      ];
      for (const sel of playSelectors) {
        const btn = await page.$(sel);
        if (btn) {
          await btn.click().catch(() => {});
          break;
        }
      }
    }
  } catch (err) {
    console.error(`page navigation failed: ${err.message}`);
    await browser.close();
    process.exit(1);
  }

  const start = Date.now();
  while (candidates.length === 0 && Date.now() - start < timeoutMs) {
    await page.waitForTimeout(250);
  }

  if (candidates.length === 0) {
    try {
      const html = await page.content();
      const match = html.match(/https?:[^"'\s]+\.m3u8[^"'\s]*/i);
      if (match && match[0]) {
        consider(match[0]);
      }
    } catch (err) {
      console.error(`fallback page content scan failed: ${err.message}`);
    }
  }

  if (candidates.length === 0) {
    console.error('no m3u8 observed in network traffic or page content');
    await browser.close();
    process.exit(1);
  }

  if (settleAfterFirstMs > 0) {
    await page.waitForTimeout(settleAfterFirstMs);
  }

  candidates.sort((a, b) => score(b) - score(a));
  const best = candidates[0];

  const cookiesForBest = await context.cookies(best).catch(() => []);
  const cookieStr = cookiesForBest
    .map((c) => `${c.name}=${c.value}`)
    .join('; ')
    .trim();

  await browser.close();

  console.log(best);
  if (cookieStr) {
    console.log(`COOKIES: ${cookieStr}`);
  }

  process.exit(0);
})().catch(async (err) => {
  console.error(`headless capture failed: ${err.message}`);
  process.exit(1);
});
