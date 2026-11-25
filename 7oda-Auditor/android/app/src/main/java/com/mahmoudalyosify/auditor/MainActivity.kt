package com.mahmoudalyosify.auditor

import android.app.Application
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mahmoudalyosify.auditor.model.AppModel
import com.mahmoudalyosify.auditor.ui.AuditorTheme
import com.mahmoudalyosify.auditor.ui.RootScreen

/** Holds the app model across configuration changes. */
class AuditorViewModel(application: Application) : AndroidViewModel(application) {
    val model: AppModel = AppModel.live(application)

    override fun onCleared() {
        model.haptics?.stop()
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AuditorTheme {
                val vm: AuditorViewModel = viewModel()
                RootScreen(vm.model)
            }
        }
    }
}
