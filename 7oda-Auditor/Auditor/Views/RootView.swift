import SwiftUI

/// First-run calibration, then the three tabs.
struct RootView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Group {
            if model.needsCalibration {
                NavigationStack {
                    CalibrationView(firstRun: true)
                }
            } else {
                TabView {
                    CheckView()
                        .tabItem { Label("Who's holding?", systemImage: "hand.raised") }
                    PeopleView()
                        .tabItem { Label("People", systemImage: "person.2") }
                    LabView()
                        .tabItem { Label("Lab", systemImage: "waveform.path.ecg") }
                }
            }
        }
        .overlay(alignment: .top) {
            if let status = model.busyStatus {
                Text(status)
                    .font(.footnote)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.thinMaterial, in: Capsule())
                    .padding(.top, 4)
            }
        }
        .task {
            if model.needsRecompute {
                await model.recomputeAll()
            }
        }
        .alert("Something went wrong", isPresented: errorShown) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var errorShown: Binding<Bool> {
        Binding(get: { model.errorMessage != nil },
                set: { shown in if !shown { model.errorMessage = nil } })
    }
}
