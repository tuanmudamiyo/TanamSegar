# TanamSegar

A web dashboard for real-time soil and climate monitoring, built as a Final Year
Project (PSM 1) at Universiti Teknikal Malaysia Melaka.

An ESP32 in the field reads soil and air conditions, pushes them to Firebase
Realtime Database, and this dashboard renders them live in the browser.

**Live site:** https://tuanmudamiyo.github.io/TanamSegar/

## What it shows

| Page | Contents |
|---|---|
| Home | Live gauges for climate, daylight, soil condition and NPK nutrients |
| Trends | History charts for all ten measurements, plus dataset export to Excel |
| Analyze | Reserved for the deep-learning analysis (in development) |
| Profile | Account details and selected plant |

Readings come from a DHT22 (air temperature and humidity), an LDR (daylight),
and an RS485 7-in-1 soil probe (moisture, temperature, EC, pH, nitrogen,
phosphorus, potassium).

## Running it

The site is static — no build step and no dependencies to install. Open
`index.html` through any web server, or visit the live link above.

Sign-in uses Firebase Authentication. Data is read from Firebase Realtime
Database; security rules make every sensor path read-only to signed-in clients,
so only the hardware can write readings.

## Stack

- Plain HTML, CSS and ES modules — no framework, no bundler
- Firebase v10.12.2 (Authentication + Realtime Database) from the Google CDN
- SheetJS for client-side `.xlsx` export
- Hand-written SVG for all gauges and charts

## Repository layout

```
index.html     Welcome splash and sign-in (entry point)
home.html      Live readings
trends.html    History charts and dataset download
analyze.html   Analysis (placeholder)
profile.html   Account and plant selection
styles.css     Shared theme tokens and components
```

The ESP32 firmware and the Python analysis service live outside this repository.
