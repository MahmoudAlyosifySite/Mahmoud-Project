// swift-tools-version:5.9
// AuditorCore: the platform-independent core of 7oda Auditor (Foundation only).
// See docs/SPEC.md for the normative description of every algorithm.

import PackageDescription

let package = Package(
    name: "AuditorCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v13),
    ],
    products: [
        .library(name: "AuditorCore", targets: ["AuditorCore"]),
    ],
    targets: [
        .target(
            name: "AuditorCore",
            path: "Sources/AuditorCore"
        ),
        .testTarget(
            name: "AuditorCoreTests",
            dependencies: ["AuditorCore"],
            path: "Tests/AuditorCoreTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
