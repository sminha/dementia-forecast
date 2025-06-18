import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Exercise {
  exerciseLogs: {
    type: number,
    startTime: string,
    endTime: string,
    duration: number,
    averageHeartRate: number,
    calorie: number,
  }[],
}

interface Walk {
  summary: {
    totalStepCount: number,
    totalDistance: number,
  },
  stepLogs: {
    startTime: string,
    endTime: string,
    stepCount: number,
    distance: number,
  }[],
}

interface Sleep {
  summary: {
    startTime: string,
    endTime: string,
    totalLightDuration: number,
    totalDeepDuration: number,
    totalRemDuration: number,
    totalDuration: number,
    efficiency: number,
  },
  sleepLogs: {
    stage: number,
    startTime: string,
    endTime: string,
  }[],
}

interface HeartRate {
  summary: {
    averageHeartRate: number,
  },
  heartRateLogs: {
    startTime: string,
    endTime: string,
    heartRate: number,
  }[],
  rmssdLogs: {
    startTime: string,
    endTime: string,
    rmssd: number,
  }[],
}

export interface BiometricDataByDate {
  height: number;
  weight: number;
  biometricData: {
    date: string;
    biometricDataList: {
      exercise: Exercise,
      walk: Walk,
      sleep: Sleep,
      heartRate: HeartRate,
    }[];
  }[];
}

const initialState: BiometricDataByDate = {
  height: 160,
  weight: 50,
  biometricData: [],
};

// export const createReport = createAsyncThunk(
//   'biometric/create',
//   async (
//     { token, biometricInfo }: { token: string, biometricInfo: BiometricDataByDate }, { rejectWithValue }
//   ) => {
//     // FOR LOCAL TEST
//     // success
//     // console.log('레포트 생성 성공');
//     // return {
//     //   'risk_score': 1,
//     //   'risk_level': 'high',
//     //   'question_list': [{ 'question_id': 1, 'answer': '1' }, { 'question_id': 2, 'answer': '2' }, { 'question_id': 3, 'answer': '3' }, { 'question_id': 4, 'answer': '4' }, { 'question_id': 5, 'answer': '5' }, { 'question_id': 6, 'answer': '6' }, { 'question_id': 7, 'answer': '7' }, { 'question_id': 8, 'answer': '8' }, { 'question_id': 9, 'answer': '9' }, { 'question_id': 10, 'answer': '10' }, { 'question_id': 11, 'answer': '11' }, { 'question_id': 12, 'answer': '12' }, { 'question_id': 13, 'answer': '13' }, { 'question_id': 14, 'answer': '14' }, { 'question_id': 15, 'answer': '15' }, { 'question_id': 16, 'answer': '16' }, { 'question_id': 17, 'answer': '17' }],
//     //   'biometric_data_list': [{ 'biometric_data_id': 1, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 2, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 3, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 4, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 5, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 6, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 7, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 8, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 9, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 10, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 11, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 12, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 13, 'biometric_data_value': 0 }, { 'biometric_data_id': 14, 'biometric_data_value': 70620 }, { 'biometric_data_id': 15, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 16, 'biometric_data_value': 30 }, { 'biometric_data_id': 17, 'biometric_data_value': 3660 }, { 'biometric_data_id': 18, 'biometric_data_value': 18180 }, { 'biometric_data_id': 19, 'biometric_data_value': 96 }, { 'biometric_data_id': 20, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 21, 'biometric_data_value': 1 }, { 'biometric_data_id': 22, 'biometric_data_value': 9960 }, { 'biometric_data_id': 23, 'biometric_data_value': 8625 }, { 'biometric_data_id': 24, 'biometric_data_value': 15825 }, { 'biometric_data_id': 25, 'biometric_data_value': 3 }, { 'biometric_data_id': 26, 'biometric_data_value': 4560 }, { 'biometric_data_id': 27, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 28, 'biometric_data_value': 18180 }, { 'biometric_data_id': 29, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 30, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 31, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 32, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 33, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 34, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 35, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 36, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 37, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 38, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 39, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 40, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 41, 'biometric_data_value': 0 }, { 'biometric_data_id': 42, 'biometric_data_value': 70620 }, { 'biometric_data_id': 43, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 44, 'biometric_data_value': 30 }, { 'biometric_data_id': 45, 'biometric_data_value': 3660 }, { 'biometric_data_id': 46, 'biometric_data_value': 18180 }, { 'biometric_data_id': 47, 'biometric_data_value': 96 }, { 'biometric_data_id': 48, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 49, 'biometric_data_value': 1 }, { 'biometric_data_id': 50, 'biometric_data_value': 9960 }, { 'biometric_data_id': 51, 'biometric_data_value': 8625 }, { 'biometric_data_id': 52, 'biometric_data_value': 15825 }, { 'biometric_data_id': 53, 'biometric_data_value': 3 }, { 'biometric_data_id': 54, 'biometric_data_value': 4560 }, { 'biometric_data_id': 55, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 56, 'biometric_data_value': 18180 }, { 'biometric_data_id': 57, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 58, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 59, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 60, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 61, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 62, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 63, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 64, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 65, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 66, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 67, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 68, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 69, 'biometric_data_value': 0 }, { 'biometric_data_id': 70, 'biometric_data_value': 70620 }, { 'biometric_data_id': 71, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 72, 'biometric_data_value': 30 }, { 'biometric_data_id': 73, 'biometric_data_value': 3660 }, { 'biometric_data_id': 74, 'biometric_data_value': 18180 }, { 'biometric_data_id': 75, 'biometric_data_value': 96 }, { 'biometric_data_id': 76, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 77, 'biometric_data_value': 1 }, { 'biometric_data_id': 78, 'biometric_data_value': 9960 }, { 'biometric_data_id': 79, 'biometric_data_value': 8625 }, { 'biometric_data_id': 80, 'biometric_data_value': 15825 }, { 'biometric_data_id': 81, 'biometric_data_value': 3 }, { 'biometric_data_id': 82, 'biometric_data_value': 4560 }, { 'biometric_data_id': 83, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 84, 'biometric_data_value': 18180 }],
//     //   'message': '레포트 생성 성공',
//     // };
//     // fail
//     console.log('레포트 생성 실패');
//     return rejectWithValue({
//       'risk_score': 1,
//       'risk_level': 'high',
//       'question_list': [{ 'question_id': 1, 'answer': '1' }, { 'question_id': 2, 'answer': '2' }, { 'question_id': 3, 'answer': '3' }, { 'question_id': 4, 'answer': '4' }, { 'question_id': 5, 'answer': '5' }, { 'question_id': 6, 'answer': '6' }, { 'question_id': 7, 'answer': '7' }, { 'question_id': 8, 'answer': '8' }, { 'question_id': 9, 'answer': '9' }, { 'question_id': 10, 'answer': '10' }, { 'question_id': 11, 'answer': '11' }, { 'question_id': 12, 'answer': '12' }, { 'question_id': 13, 'answer': '13' }, { 'question_id': 14, 'answer': '14' }, { 'question_id': 15, 'answer': '15' }, { 'question_id': 16, 'answer': '16' }, { 'question_id': 17, 'answer': '17' }],
//       'biometric_data_list': [{ 'biometric_data_id': 1, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 2, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 3, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 4, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 5, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 6, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 7, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 8, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 9, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 10, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 11, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 12, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 13, 'biometric_data_value': 0 }, { 'biometric_data_id': 14, 'biometric_data_value': 70620 }, { 'biometric_data_id': 15, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 16, 'biometric_data_value': 30 }, { 'biometric_data_id': 17, 'biometric_data_value': 3660 }, { 'biometric_data_id': 18, 'biometric_data_value': 18180 }, { 'biometric_data_id': 19, 'biometric_data_value': 96 }, { 'biometric_data_id': 20, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 21, 'biometric_data_value': 1 }, { 'biometric_data_id': 22, 'biometric_data_value': 9960 }, { 'biometric_data_id': 23, 'biometric_data_value': 8625 }, { 'biometric_data_id': 24, 'biometric_data_value': 15825 }, { 'biometric_data_id': 25, 'biometric_data_value': 3 }, { 'biometric_data_id': 26, 'biometric_data_value': 4560 }, { 'biometric_data_id': 27, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 28, 'biometric_data_value': 18180 }, { 'biometric_data_id': 29, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 30, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 31, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 32, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 33, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 34, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 35, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 36, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 37, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 38, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 39, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 40, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 41, 'biometric_data_value': 0 }, { 'biometric_data_id': 42, 'biometric_data_value': 70620 }, { 'biometric_data_id': 43, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 44, 'biometric_data_value': 30 }, { 'biometric_data_id': 45, 'biometric_data_value': 3660 }, { 'biometric_data_id': 46, 'biometric_data_value': 18180 }, { 'biometric_data_id': 47, 'biometric_data_value': 96 }, { 'biometric_data_id': 48, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 49, 'biometric_data_value': 1 }, { 'biometric_data_id': 50, 'biometric_data_value': 9960 }, { 'biometric_data_id': 51, 'biometric_data_value': 8625 }, { 'biometric_data_id': 52, 'biometric_data_value': 15825 }, { 'biometric_data_id': 53, 'biometric_data_value': 3 }, { 'biometric_data_id': 54, 'biometric_data_value': 4560 }, { 'biometric_data_id': 55, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 56, 'biometric_data_value': 18180 }, { 'biometric_data_id': 57, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 58, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 59, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 60, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 61, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 62, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 63, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 64, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 65, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 66, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 67, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 68, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 69, 'biometric_data_value': 0 }, { 'biometric_data_id': 70, 'biometric_data_value': 70620 }, { 'biometric_data_id': 71, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 72, 'biometric_data_value': 30 }, { 'biometric_data_id': 73, 'biometric_data_value': 3660 }, { 'biometric_data_id': 74, 'biometric_data_value': 18180 }, { 'biometric_data_id': 75, 'biometric_data_value': 96 }, { 'biometric_data_id': 76, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 77, 'biometric_data_value': 1 }, { 'biometric_data_id': 78, 'biometric_data_value': 9960 }, { 'biometric_data_id': 79, 'biometric_data_value': 8625 }, { 'biometric_data_id': 80, 'biometric_data_value': 15825 }, { 'biometric_data_id': 81, 'biometric_data_value': 3 }, { 'biometric_data_id': 82, 'biometric_data_value': 4560 }, { 'biometric_data_id': 83, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 84, 'biometric_data_value': 18180 }],
//       'message': '레포트 생성 성공',
//     }.message);

//     // FOR SERVER COMMUNICATION
//     // try {
//     //   const response = await fetch('BACKEND_URL/prediction/report/create', {
//     //     method: 'POST',
//     //     headers: {
//     //       Authorization: `Bearer ${token}`,
//     //     },
//     //     body: JSON.stringify(biometricInfo),
//     //   });

//     //   const data = await response.json();

//     //   if (!response.ok) {
//     //     return rejectWithValue(data.message || '레포트 생성 실패');
//     //   }

//     //   return data;
//     // } catch (error) {
//     //   return rejectWithValue('알 수 없는 오류가 발생했습니다.');
//     // }
//   }
// );

const biometricSlice = createSlice({
  name: 'biometric',
  initialState,
  reducers: {
    setBiometricInfo: (state, action: PayloadAction<{ field: keyof BiometricDataByDate, value: any }>) => {
      (state[action.payload.field] as typeof action.payload.value) = action.payload.value;
    },
  },
  // extraReducers: (builder) => {
  //   builder
  //     .addCase(createReport.pending, (state) => {

  //     })
  //     .addCase(createReport.fulfilled, (state, action) => {

  //     })
  //     .addCase(createReport.rejected, (state, action) => {

  //     });
  // },
});

export const { setBiometricInfo } = biometricSlice.actions;
export default biometricSlice.reducer;