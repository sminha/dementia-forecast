package com.dementiaforecast

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

class PermissionsRationaleActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val textView = TextView(this).apply {
            text = "이 앱은 건강 데이터를 사용하여 치매 예측 기능을 제공합니다.\n\n데이터 접근 권한이 필요해요 🙏"
            textSize = 16f
            setPadding(40, 100, 40, 100)
        }

        setContentView(textView)
    }
}
