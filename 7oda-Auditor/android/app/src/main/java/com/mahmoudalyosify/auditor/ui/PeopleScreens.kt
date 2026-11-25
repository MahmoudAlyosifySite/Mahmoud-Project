package com.mahmoudalyosify.auditor.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.unit.dp
import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.core.Hand
import com.mahmoudalyosify.auditor.model.AppModel
import com.mahmoudalyosify.auditor.model.Person
import com.mahmoudalyosify.auditor.model.PrototypeCount

fun Accessory.title(): String = rawValue.replaceFirstChar { it.uppercase() }

@Composable
fun PeopleScreen(model: AppModel, onOpen: (String) -> Unit, onAdd: () -> Unit) {
    Box(Modifier.fillMaxSize()) {
        LazyColumn(Modifier.fillMaxSize()) {
            if (model.people.isEmpty()) {
                item {
                    Text("Nobody is enrolled yet. Tap Add person to start.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(16.dp))
                }
            }
            items(model.people, key = { it.id }) { person ->
                PersonRow(model, person, Modifier.clickable { onOpen(person.id) })
                HorizontalDivider()
            }
        }
        ExtendedFloatingActionButton(
            onClick = onAdd,
            icon = { Icon(Icons.Filled.Add, contentDescription = null) },
            text = { Text("Add person") },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(16.dp),
        )
    }
}

private fun countSummary(counts: List<PrototypeCount>): String {
    fun count(hand: Hand) = counts.filter { it.key.hand == hand }.sumOf { it.v0 }
    return "Right ${count(Hand.right)} · Left ${count(Hand.left)} samples"
}

@Composable
private fun PersonRow(model: AppModel, person: Person, modifier: Modifier) {
    val counts = model.prototypeCounts(person.id)
    Column(
        modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(person.name, style = MaterialTheme.typography.titleMedium)
        FootnoteText(countSummary(counts.filter { it.key.accessory == Accessory.none }))
        for (acc in Accessory.entries.filter { it != Accessory.none }) {
            val group = counts.filter { it.key.accessory == acc }
            if (group.isNotEmpty()) FootnoteText("${acc.title()}: ${countSummary(group)}")
        }
    }
}

@Composable
fun PersonDetailScreen(
    model: AppModel,
    personID: String,
    onReenroll: () -> Unit,
    onAccessory: (Accessory) -> Unit,
    onDeleted: () -> Unit,
) {
    var name by remember(personID) { mutableStateOf(model.personName(personID)) }
    var accessory by remember { mutableStateOf(Accessory.watch) }
    var menuOpen by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SectionCard("Name") {
            OutlinedTextField(
                value = name, onValueChange = { name = it }, singleLine = true, label = { Text("Name") },
                keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words),
                modifier = Modifier.fillMaxWidth(),
            )
            Button(
                onClick = { model.rename(personID, name) },
                enabled = name.isNotBlank() && name.trim() != model.personName(personID),
            ) { Text("Save name") }
        }
        SectionCard("Samples (v0 / v1)") {
            val counts = model.prototypeCounts(personID)
            if (counts.isEmpty()) FootnoteText("No samples")
            for (c in counts) {
                val hand = if (c.key.hand == Hand.right) "Right hand" else "Left hand"
                val label = if (c.key.accessory == Accessory.none) hand else "$hand · ${c.key.accessory.rawValue}"
                LabeledRow(label, "${c.v0} / ${c.v1}")
            }
        }
        SectionCard("Accessory") {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Box {
                    OutlinedButton(onClick = { menuOpen = true }) { Text(accessory.title()) }
                    DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                        for (acc in Accessory.entries.filter { it != Accessory.none }) {
                            DropdownMenuItem(text = { Text(acc.title()) }, onClick = {
                                accessory = acc
                                menuOpen = false
                            })
                        }
                    }
                }
                Button(onClick = { onAccessory(accessory) }) { Text("Add accessory") }
            }
            FootnoteText("Records 5 patterns per hand while wearing the accessory, so the app also knows you with it on.")
        }
        SectionCard {
            Button(onClick = onReenroll, modifier = Modifier.fillMaxWidth()) { Text("Re-enroll") }
            OutlinedButton(onClick = { confirmDelete = true }, modifier = Modifier.fillMaxWidth()) {
                Text("Delete person", color = MaterialTheme.colorScheme.error)
            }
            FootnoteText("Re-enrolling replaces the samples recorded without accessories.")
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete ${model.personName(personID)} and all their recordings?") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    model.deletePerson(personID)
                    onDeleted()
                }) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
        )
    }
}
