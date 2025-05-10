package com.dementiaforecast

import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.fabricEnabled
import com.facebook.react.defaults.DefaultReactActivityDelegate

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
// import android.util.Log
// import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
// import androidx.activity.result.ActivityResultLauncher
// import androidx.lifecycle.lifecycleScope
// import kotlinx.coroutines.launch
import androidx.health.connect.client.HealthConnectClient
// import androidx.health.connect.client.errors.HealthConnectClientException
// import androidx.health.connect.client.permission.HealthPermission
// import androidx.health.connect.client.permission.PermissionController
// import androidx.health.connect.client.records.*

class MainActivity : ReactActivity() {

  /**
   * Returns the name of the main component registered from JavaScript. This is used to schedule
   * rendering of the component.
   */
  override fun getMainComponentName(): String = "DementiaForecast"

  /**
   * Returns the instance of the [ReactActivityDelegate]. We use [DefaultReactActivityDelegate]
   * which allows you to enable New Architecture with a single boolean flags [fabricEnabled]
   */
  override fun createReactActivityDelegate(): ReactActivityDelegate =
      DefaultReactActivityDelegate(this, mainComponentName, fabricEnabled)
  
  // private lateinit var requestPermissions: ActivityResultLauncher<Set<String>>

  // 추가: onCreate 메서드 오버라이드하여 헬스 커넥트 상태 확인
  override fun onCreate(savedInstanceState: Bundle?) {
      super.onCreate(savedInstanceState)

  //     // 로그 추가: onCreate가 호출되었음을 확인
  //     Log.d("MainActivity", "onCreate called")

      // 헬스 커넥트 패키지 이름
      val providerPackageName = "com.google.android.apps.healthdata"
      
      // HealthConnectClient 상태 확인
      val availabilityStatus = HealthConnectClient.getSdkStatus(applicationContext, providerPackageName)

      // 헬스 커넥트가 설치되지 않은 경우 처리
      if (availabilityStatus == HealthConnectClient.SDK_UNAVAILABLE) {
  //       Log.d("MainActivity", "Health Connect SDK is unavailable") // 로그 추가
        return
      }

      // 헬스 커넥트 제공자 업데이트가 필요한 경우 처리
      if (availabilityStatus == HealthConnectClient.SDK_UNAVAILABLE_PROVIDER_UPDATE_REQUIRED) {
  //       Log.d("MainActivity", "Health Connect SDK requires update") // 로그 추가
        val uriString = "market://details?id=$providerPackageName&url=healthconnect%3A%2F%2Fonboarding"
        startActivity(
          Intent(Intent.ACTION_VIEW).apply {
            setPackage("com.android.vending")
            data = Uri.parse(uriString)
            putExtra("overlay", true)
            putExtra("callerId", packageName)
          }
        )
        return
      }

      // 헬스 커넥트 클라이언트 인스턴스 가져오기
      val healthConnectClient = HealthConnectClient.getOrCreate(applicationContext)
  //     Log.d("MainActivity", "Health Connect Client initialized") // 로그 추가
  //     // 이제 healthConnectClient를 사용하여 데이터 작업 진행 가능

  //     // 요청할 권한 집합 정의
  //     val PERMISSIONS = setOf(
  //         // 심박수
  //         HealthPermission.getReadPermission(HeartRateRecord::class),

  //         // 걸음수
  //         HealthPermission.getReadPermission(StepsRecord::class),

  //         // 이동거리
  //         HealthPermission.getReadPermission(DistanceRecord::class),

  //         // 수면 세션
  //         HealthPermission.getReadPermission(SleepSessionRecord::class),

  //         // 운동 세션 (활동)
  //         HealthPermission.getReadPermission(ExerciseSessionRecord::class),

  //         // 휴식 심박수
  //         HealthPermission.getReadPermission(RestingHeartRateRecord::class),

  //         // 기초대사율
  //         HealthPermission.getReadPermission(BasalMetabolicRateRecord::class),

  //         // 활동 칼로리
  //         HealthPermission.getReadPermission(ActiveCaloriesBurnedRecord::class),

  //         // 총 칼로리
  //         HealthPermission.getReadPermission(TotalCaloriesBurnedRecord::class)
  //     )

  //     // 권한 요청 런처 초기화
  //     requestPermissions = registerForActivityResult(
  //       PermissionController.createRequestPermissionResultContract()
  //     ) { granted ->
  //       if (granted.containsAll(PERMISSIONS)) {
  //         // 모든 권한 허용됨
  //         Log.d("MainActivity", "All permissions granted") // 로그 추가
  //       } else {
  //         // 일부 권한이 거부됨
  //         Log.d("MainActivity", "Some permissions denied") // 로그 추가
  //       }
  //     }

  //     // 권한 체크 및 요청
  //     lifecycleScope.launch {
  //       val granted = healthConnectClient.permissionController.getGrantedPermissions()
  //       if (!granted.containsAll(PERMISSIONS)) {
  //         Log.d("MainActivity", "Permissions not granted, requesting...") // 로그 추가
  //         requestPermissions.launch(PERMISSIONS)
  //       } else {
  //         // 이미 모든 권한이 허용된 상태
  //         Log.d("MainActivity", "Permissions already granted") // 로그 추가
  //       }
  //     }
  }
}