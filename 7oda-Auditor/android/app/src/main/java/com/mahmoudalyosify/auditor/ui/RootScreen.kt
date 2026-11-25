@file:OptIn(ExperimentalMaterial3Api::class)

package com.mahmoudalyosify.auditor.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.PanTool
import androidx.compose.material.icons.filled.People
import androidx.compose.material.icons.filled.Science
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.mahmoudalyosify.auditor.core.Accessory
import com.mahmoudalyosify.auditor.model.AppModel
import com.mahmoudalyosify.auditor.model.EnrollmentKind

/** Navigation routes. */
object Routes {
    const val CHECK = "check"
    const val PEOPLE = "people"
    const val LAB = "lab"
    const val PERSON = "person/{id}"
    const val ENROLL_NEW = "enroll/new"
    const val ENROLL_AGAIN = "enroll/again/{id}"
    const val ENROLL_ACCESSORY = "enroll/accessory/{id}/{accessory}"
    const val TEST_RUN = "lab/test"
    const val SIGNAL_CHECK = "lab/signal"
    const val SETTINGS = "lab/settings"
    const val CALIBRATE = "lab/calibrate"

    fun person(id: String) = "person/$id"
    fun enrollAgain(id: String) = "enroll/again/$id"
    fun enrollAccessory(id: String, accessory: Accessory) = "enroll/accessory/$id/${accessory.rawValue}"
}

private data class Tab(val route: String, val label: String, val icon: ImageVector)

private val tabs = listOf(
    Tab(Routes.CHECK, "Who's holding?", Icons.Filled.PanTool),
    Tab(Routes.PEOPLE, "People", Icons.Filled.People),
    Tab(Routes.LAB, "Lab", Icons.Filled.Science),
)

@Composable
fun RootScreen(model: AppModel) {
    if (model.needsCalibration) {
        Scaffold(topBar = { TopAppBar(title = { Text("Calibrate") }) }) { padding ->
            CalibrationScreen(model, firstRun = true, modifier = Modifier.padding(padding))
        }
    } else {
        MainTabs(model)
    }

    LaunchedEffect(Unit) {
        if (model.needsRecompute) model.recomputeAll()
    }

    model.busyStatus?.let { status ->
        Box(Modifier.statusBarsPadding().padding(top = 4.dp), contentAlignment = Alignment.TopCenter) {
            Text(
                status, style = MaterialTheme.typography.labelMedium,
                modifier = Modifier
                    .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(50))
                    .padding(horizontal = 12.dp, vertical = 6.dp),
            )
        }
    }

    model.errorMessage?.let { message ->
        AlertDialog(
            onDismissRequest = { model.errorMessage = null },
            confirmButton = { TextButton(onClick = { model.errorMessage = null }) { Text("OK") } },
            title = { Text("Something went wrong") },
            text = { Text(message) },
        )
    }
}

@Composable
private fun MainTabs(model: AppModel) {
    val nav = rememberNavController()
    val entry by nav.currentBackStackEntryAsState()
    val route = entry?.destination?.route
    val topLevel = tabs.any { it.route == route } || route == null
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(titleFor(route, model, entry?.arguments?.getString("id"))) },
                navigationIcon = {
                    if (!topLevel) {
                        IconButton(onClick = { nav.popBackStack() }) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                        }
                    }
                },
            )
        },
        bottomBar = {
            if (topLevel) {
                NavigationBar {
                    for (tab in tabs) {
                        NavigationBarItem(
                            selected = route == tab.route,
                            onClick = {
                                nav.navigate(tab.route) {
                                    popUpTo(nav.graph.findStartDestination().id) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = { Icon(tab.icon, contentDescription = null) },
                            label = { Text(tab.label) },
                        )
                    }
                }
            }
        },
    ) { padding ->
        NavHost(nav, startDestination = Routes.CHECK, modifier = Modifier.padding(padding)) {
            composable(Routes.CHECK) { CheckScreen(model) }
            composable(Routes.PEOPLE) {
                PeopleScreen(model, onOpen = { nav.navigate(Routes.person(it)) },
                    onAdd = { nav.navigate(Routes.ENROLL_NEW) })
            }
            composable(Routes.LAB) { LabScreen(model, nav) }
            composable(Routes.PERSON, arguments = listOf(navArgument("id") { type = NavType.StringType })) { e ->
                val id = e.arguments?.getString("id") ?: ""
                PersonDetailScreen(
                    model, id,
                    onReenroll = { nav.navigate(Routes.enrollAgain(id)) },
                    onAccessory = { acc -> nav.navigate(Routes.enrollAccessory(id, acc)) },
                    onDeleted = { nav.popBackStack() },
                )
            }
            composable(Routes.ENROLL_NEW) {
                EnrollmentScreen(model, EnrollmentKind.NewPerson, onDone = { nav.popBackStack() })
            }
            composable(Routes.ENROLL_AGAIN, arguments = listOf(navArgument("id") { type = NavType.StringType })) { e ->
                val id = e.arguments?.getString("id") ?: ""
                EnrollmentScreen(model, EnrollmentKind.Reenroll(id, model.personName(id)), onDone = { nav.popBackStack() })
            }
            composable(
                Routes.ENROLL_ACCESSORY,
                arguments = listOf(
                    navArgument("id") { type = NavType.StringType },
                    navArgument("accessory") { type = NavType.StringType },
                ),
            ) { e ->
                val id = e.arguments?.getString("id") ?: ""
                val accessory = Accessory.from(e.arguments?.getString("accessory") ?: "") ?: Accessory.other
                EnrollmentScreen(model, EnrollmentKind.AccessoryEnroll(id, model.personName(id), accessory),
                    onDone = { nav.popBackStack() })
            }
            composable(Routes.TEST_RUN) { TestRunScreen(model) }
            composable(Routes.SIGNAL_CHECK) { SignalCheckScreen(model) }
            composable(Routes.SETTINGS) { LabSettingsScreen(model) }
            composable(Routes.CALIBRATE) { CalibrationScreen(model, firstRun = false) }
        }
    }
}

private fun titleFor(route: String?, model: AppModel, id: String?): String = when (route) {
    Routes.PEOPLE -> "People"
    Routes.LAB -> "Lab"
    Routes.PERSON -> model.personName(id)
    Routes.ENROLL_NEW -> "Add person"
    Routes.ENROLL_AGAIN -> "Re-enroll ${model.personName(id)}"
    Routes.ENROLL_ACCESSORY -> "${model.personName(id)} · accessory"
    Routes.TEST_RUN -> "Test run"
    Routes.SIGNAL_CHECK -> "Signal check"
    Routes.SETTINGS -> "Settings"
    Routes.CALIBRATE -> "Calibrate"
    else -> "Who's holding?"
}

/** Navigates within the Lab (exposed for the Lab screen). */
fun NavHostController.openLab(route: String) = navigate(route)
