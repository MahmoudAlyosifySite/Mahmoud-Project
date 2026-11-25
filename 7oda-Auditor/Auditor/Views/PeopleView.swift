import SwiftUI
import AuditorCore

/// Wraps an enrollment kind for `.sheet(item:)`.
struct EnrollmentRequest: Identifiable {
    let id = UUID()
    let kind: EnrollmentKind
}

/// People: sample counts per hand and accessory prototypes; add, open details.
struct PeopleView: View {
    @Environment(AppModel.self) private var model
    @State private var request: EnrollmentRequest?

    var body: some View {
        NavigationStack {
            List {
                if model.people.isEmpty {
                    Text("Nobody is enrolled yet. Tap + to add a person.")
                        .foregroundStyle(.secondary)
                }
                ForEach(model.people) { person in
                    NavigationLink(value: person) {
                        PersonRow(person: person)
                    }
                }
            }
            .navigationTitle("People")
            .navigationDestination(for: Person.self) { person in
                PersonDetailView(personID: person.id)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        request = EnrollmentRequest(kind: .newPerson)
                    } label: {
                        Label("Add person", systemImage: "plus")
                    }
                }
            }
            .sheet(item: $request) { req in
                EnrollmentView(kind: req.kind)
            }
        }
    }
}

struct PersonRow: View {
    @Environment(AppModel.self) private var model
    let person: Person

    var body: some View {
        let counts = model.prototypeCounts(for: person.id)
        VStack(alignment: .leading, spacing: 4) {
            Text(person.name).font(.headline)
            Text(summary(counts.filter { $0.key.accessory == Accessory.none }))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            ForEach(Accessory.allCases.filter { $0 != Accessory.none }, id: \.self) { acc in
                let group = counts.filter { $0.key.accessory == acc }
                if !group.isEmpty {
                    Text("\(acc.rawValue.capitalized): \(summary(group))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func summary(_ counts: [PrototypeCount]) -> String {
        func count(_ hand: Hand) -> Int { counts.filter { $0.key.hand == hand }.reduce(0) { $0 + $1.v0 } }
        return "Right \(count(.right)) · Left \(count(.left)) samples"
    }
}

/// Person detail: rename, add accessory (5 patterns per hand), re-enroll, delete.
struct PersonDetailView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    let personID: String

    @State private var name = ""
    @State private var accessory: Accessory = .watch
    @State private var request: EnrollmentRequest?
    @State private var confirmDelete = false

    var body: some View {
        Form {
            Section("Name") {
                TextField("Name", text: $name)
                    .textInputAutocapitalization(.words)
                Button("Save name") { model.rename(personID: personID, to: name) }
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty || name == model.personName(personID))
            }

            Section("Samples (v0 / v1)") {
                let counts = model.prototypeCounts(for: personID)
                if counts.isEmpty {
                    Text("No samples").foregroundStyle(.secondary)
                }
                ForEach(counts) { c in
                    LabeledContent(label(c.key), value: "\(c.v0) / \(c.v1)")
                }
            }

            Section {
                Picker("Accessory", selection: $accessory) {
                    ForEach(Accessory.allCases.filter { $0 != Accessory.none }, id: \.self) { acc in
                        Text(acc.rawValue.capitalized).tag(acc)
                    }
                }
                Button("Add accessory") {
                    request = EnrollmentRequest(kind: .accessory(personID: personID, name: model.personName(personID),
                                                                 accessory: accessory))
                }
            } header: {
                Text("Accessory")
            } footer: {
                Text("Records 5 patterns per hand while wearing the accessory, so the app also knows you with it on.")
            }

            Section {
                Button("Re-enroll") {
                    request = EnrollmentRequest(kind: .reenroll(personID: personID, name: model.personName(personID)))
                }
                Button("Delete person", role: .destructive) { confirmDelete = true }
            } footer: {
                Text("Re-enrolling replaces the samples recorded without accessories.")
            }
        }
        .navigationTitle(model.personName(personID))
        .onAppear { name = model.personName(personID) }
        .sheet(item: $request) { req in
            EnrollmentView(kind: req.kind)
        }
        .confirmationDialog("Delete \(model.personName(personID)) and all their recordings?",
                            isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                model.deletePerson(personID)
                dismiss()
            }
        }
    }

    private func label(_ key: PrototypeKey) -> String {
        let hand = key.hand == .right ? "Right hand" : "Left hand"
        return key.accessory == Accessory.none ? hand : "\(hand) · \(key.accessory.rawValue)"
    }
}
