# Installing 7oda Auditor without a Mac

Every push to `main` that passes CI publishes the install files on the pre-release
[`7oda-auditor-latest`](https://github.com/MahmoudAlyosifySite/Mahmoud-Project/releases/tag/7oda-auditor-latest):

| File | Phone |
|---|---|
| `7oda-Auditor-unsigned.ipa` | iPhone 8 or newer, iOS 17+ |
| `7oda-Auditor.apk` | Android 8.0+ (once the Android app is in CI) |

## iPhone from a Windows PC (free, with Sideloadly)

The `.ipa` is unsigned. Sideloadly signs it with your own Apple ID, the same free mechanism Xcode uses for a personal team. Apps signed this way run for 7 days; refreshing keeps the data.

1. **Prepare Windows.** Install iTunes and iCloud from apple.com (the downloads from Apple's website, not the Microsoft Store versions). Sideloadly needs their Apple device drivers.
2. **Install Sideloadly** from [sideloadly.io](https://sideloadly.io).
3. **Download** `7oda-Auditor-unsigned.ipa` from the release page above.
4. **Connect the iPhone** with a USB cable, unlock it and tap **Trust This Computer**.
5. **Sign and install.** Open Sideloadly, drag the `.ipa` onto it, choose your iPhone, type your Apple ID email and press **Start**. Enter your Apple ID password and the two-factor code when asked. (A second Apple ID also works if you prefer not to use your main one.)
6. **Trust the developer.** On the iPhone: *Settings → General → VPN & Device Management →* your Apple ID *→ Trust*.
7. **Turn on Developer Mode** (iOS 16+): *Settings → Privacy & Security → Developer Mode → On*; the phone restarts; confirm **Turn On**.
8. **Open 7oda Auditor**: calibrate with the phone flat on a table, then *People → Add person*.
9. **Every 7 days**: connect the phone and run Sideloadly again with the same `.ipa` (or the newest one). Your people and recordings stay on the phone.

Limits of a free Apple ID: at most 3 sideloaded apps at a time and 10 app IDs per week.

## iPhone with TestFlight (paid, optional)

With an Apple Developer Program membership the CI can sign the app and upload it to TestFlight, so it installs like a normal app and does not expire after 7 days. This needs an App Store Connect API key stored as repository secrets; it is not set up yet.

## iPhone from a Mac

Open `7oda-Auditor/Auditor.xcodeproj` in Xcode 16+, choose your Team under *Signing & Capabilities*, pick the phone and press Run.

## Android

Download `7oda-Auditor.apk` on the phone, open it, allow installs from this source when Android asks, and install. Updates published later install over the old version and keep the data (all builds use the same signing key).
