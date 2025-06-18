import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { BiometricDataByDate } from './biometricSlice.ts';
import RNFS from 'react-native-fs';

interface QuestionAnswer {
  question_id: number;
  answer: string;
}

interface BiometricDataItem {
  biometric_data_id: number;
  biometric_data_value: number;
}

interface ReportState {
  loading: boolean;
  error: string | null;
  riskScore: number | null;
  riskLevel: string | null;
  questionList: QuestionAnswer[];
  biometricDataList: BiometricDataItem[];
}

const initialState: ReportState = {
  loading: false,
  error: null,
  riskScore: null,
  riskLevel: null,
  questionList: [],
  biometricDataList: [],
};

const saveResultsToFile = async (results: any) => {
  const path = `${RNFS.DownloadDirectoryPath}/report_create_api_request_body.json`;
  try {
    await RNFS.writeFile(path, JSON.stringify(results), 'utf8');
    console.log('결과 저장 성공:', path);
  } catch (e) {
    console.log('저장 실패:', e);
  }
};

export const createReport = createAsyncThunk(
  'report/createReport',
  async (
    { token, biometricInfo }: { token: string; biometricInfo: BiometricDataByDate },
    { rejectWithValue }
  ) => {
    // FOR LOCAL TEST
    // success
    console.log('레포트 생성 성공');
    return {
      'risk_score': 40,
      'risk_level': 'low',
      'question_list': [{ 'question_id': 1, 'answer': '1' }, { 'question_id': 2, 'answer': '2' }, { 'question_id': 3, 'answer': '3' }, { 'question_id': 4, 'answer': '4' }, { 'question_id': 5, 'answer': '2' }, { 'question_id': 6, 'answer': '6' }, { 'question_id': 7, 'answer': '300' }, { 'question_id': 8, 'answer': '120' }, { 'question_id': 9, 'answer': '2' }, { 'question_id': 10, 'answer': '1' }, { 'question_id': 11, 'answer': '11' }, { 'question_id': 12, 'answer': '12' }, { 'question_id': 13, 'answer': '10' }, { 'question_id': 14, 'answer': '7' }, { 'question_id': 15, 'answer': '40' }, { 'question_id': 16, 'answer': '16' }, { 'question_id': 17, 'answer': '17' }],
      'biometric_data_list': [{ 'biometric_data_id': 1, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 2, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 3, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 4, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 5, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 6, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 7, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 8, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 9, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 10, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 11, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 12, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 13, 'biometric_data_value': 0 }, { 'biometric_data_id': 14, 'biometric_data_value': 70620 }, { 'biometric_data_id': 15, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 16, 'biometric_data_value': 30 }, { 'biometric_data_id': 17, 'biometric_data_value': 3660 }, { 'biometric_data_id': 18, 'biometric_data_value': 18180 }, { 'biometric_data_id': 19, 'biometric_data_value': 96 }, { 'biometric_data_id': 20, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 21, 'biometric_data_value': 1 }, { 'biometric_data_id': 22, 'biometric_data_value': 9960 }, { 'biometric_data_id': 23, 'biometric_data_value': 8625 }, { 'biometric_data_id': 24, 'biometric_data_value': 15825 }, { 'biometric_data_id': 25, 'biometric_data_value': 3 }, { 'biometric_data_id': 26, 'biometric_data_value': 4560 }, { 'biometric_data_id': 27, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 28, 'biometric_data_value': 18180 }, { 'biometric_data_id': 29, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 30, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 31, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 32, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 33, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 34, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 35, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 36, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 37, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 38, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 39, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 40, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 41, 'biometric_data_value': 0 }, { 'biometric_data_id': 42, 'biometric_data_value': 70620 }, { 'biometric_data_id': 43, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 44, 'biometric_data_value': 30 }, { 'biometric_data_id': 45, 'biometric_data_value': 3660 }, { 'biometric_data_id': 46, 'biometric_data_value': 18180 }, { 'biometric_data_id': 47, 'biometric_data_value': 96 }, { 'biometric_data_id': 48, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 49, 'biometric_data_value': 1 }, { 'biometric_data_id': 50, 'biometric_data_value': 9960 }, { 'biometric_data_id': 51, 'biometric_data_value': 8625 }, { 'biometric_data_id': 52, 'biometric_data_value': 15825 }, { 'biometric_data_id': 53, 'biometric_data_value': 3 }, { 'biometric_data_id': 54, 'biometric_data_value': 4560 }, { 'biometric_data_id': 55, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 56, 'biometric_data_value': 18180 }, { 'biometric_data_id': 57, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 58, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 59, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 60, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 61, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 62, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 63, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 64, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 65, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 66, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 67, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 68, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 69, 'biometric_data_value': 0 }, { 'biometric_data_id': 70, 'biometric_data_value': 70620 }, { 'biometric_data_id': 71, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 72, 'biometric_data_value': 30 }, { 'biometric_data_id': 73, 'biometric_data_value': 3660 }, { 'biometric_data_id': 74, 'biometric_data_value': 18180 }, { 'biometric_data_id': 75, 'biometric_data_value': 96 }, { 'biometric_data_id': 76, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 77, 'biometric_data_value': 1 }, { 'biometric_data_id': 78, 'biometric_data_value': 9960 }, { 'biometric_data_id': 79, 'biometric_data_value': 8625 }, { 'biometric_data_id': 80, 'biometric_data_value': 15825 }, { 'biometric_data_id': 81, 'biometric_data_value': 3 }, { 'biometric_data_id': 82, 'biometric_data_value': 4560 }, { 'biometric_data_id': 83, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 84, 'biometric_data_value': 18180 }],
      'message': '레포트 생성 성공',
    };
    // fail
    // console.log('레포트 생성 실패');
    // return rejectWithValue({
    //   'risk_score': 60,
    //   'risk_level': 'high',
    //   'question_list': [{ 'question_id': 1, 'answer': '1' }, { 'question_id': 2, 'answer': '2' }, { 'question_id': 3, 'answer': '3' }, { 'question_id': 4, 'answer': '4' }, { 'question_id': 5, 'answer': '5' }, { 'question_id': 6, 'answer': '6' }, { 'question_id': 7, 'answer': '7' }, { 'question_id': 8, 'answer': '8' }, { 'question_id': 9, 'answer': '9' }, { 'question_id': 10, 'answer': '10' }, { 'question_id': 11, 'answer': '11' }, { 'question_id': 12, 'answer': '12' }, { 'question_id': 13, 'answer': '13' }, { 'question_id': 14, 'answer': '14' }, { 'question_id': 15, 'answer': '15' }, { 'question_id': 16, 'answer': '16' }, { 'question_id': 17, 'answer': '17' }],
    //   'biometric_data_list': [{ 'biometric_data_id': 1, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 2, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 3, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 4, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 5, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 6, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 7, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 8, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 9, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 10, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 11, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 12, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 13, 'biometric_data_value': 0 }, { 'biometric_data_id': 14, 'biometric_data_value': 70620 }, { 'biometric_data_id': 15, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 16, 'biometric_data_value': 30 }, { 'biometric_data_id': 17, 'biometric_data_value': 3660 }, { 'biometric_data_id': 18, 'biometric_data_value': 18180 }, { 'biometric_data_id': 19, 'biometric_data_value': 96 }, { 'biometric_data_id': 20, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 21, 'biometric_data_value': 1 }, { 'biometric_data_id': 22, 'biometric_data_value': 9960 }, { 'biometric_data_id': 23, 'biometric_data_value': 8625 }, { 'biometric_data_id': 24, 'biometric_data_value': 15825 }, { 'biometric_data_id': 25, 'biometric_data_value': 3 }, { 'biometric_data_id': 26, 'biometric_data_value': 4560 }, { 'biometric_data_id': 27, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 28, 'biometric_data_value': 18180 }, { 'biometric_data_id': 29, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 30, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 31, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 32, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 33, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 34, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 35, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 36, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 37, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 38, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 39, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 40, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 41, 'biometric_data_value': 0 }, { 'biometric_data_id': 42, 'biometric_data_value': 70620 }, { 'biometric_data_id': 43, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 44, 'biometric_data_value': 30 }, { 'biometric_data_id': 45, 'biometric_data_value': 3660 }, { 'biometric_data_id': 46, 'biometric_data_value': 18180 }, { 'biometric_data_id': 47, 'biometric_data_value': 96 }, { 'biometric_data_id': 48, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 49, 'biometric_data_value': 1 }, { 'biometric_data_id': 50, 'biometric_data_value': 9960 }, { 'biometric_data_id': 51, 'biometric_data_value': 8625 }, { 'biometric_data_id': 52, 'biometric_data_value': 15825 }, { 'biometric_data_id': 53, 'biometric_data_value': 3 }, { 'biometric_data_id': 54, 'biometric_data_value': 4560 }, { 'biometric_data_id': 55, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 56, 'biometric_data_value': 18180 }, { 'biometric_data_id': 57, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 58, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 59, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 60, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 61, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 62, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 63, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 64, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 65, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 66, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 67, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 68, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 69, 'biometric_data_value': 0 }, { 'biometric_data_id': 70, 'biometric_data_value': 70620 }, { 'biometric_data_id': 71, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 72, 'biometric_data_value': 30 }, { 'biometric_data_id': 73, 'biometric_data_value': 3660 }, { 'biometric_data_id': 74, 'biometric_data_value': 18180 }, { 'biometric_data_id': 75, 'biometric_data_value': 96 }, { 'biometric_data_id': 76, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 77, 'biometric_data_value': 1 }, { 'biometric_data_id': 78, 'biometric_data_value': 9960 }, { 'biometric_data_id': 79, 'biometric_data_value': 8625 }, { 'biometric_data_id': 80, 'biometric_data_value': 15825 }, { 'biometric_data_id': 81, 'biometric_data_value': 3 }, { 'biometric_data_id': 82, 'biometric_data_value': 4560 }, { 'biometric_data_id': 83, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 84, 'biometric_data_value': 18180 }],
    //   'message': '레포트 생성 실패',
    // }.message);

    // FOR SERVER COMMUNICATION
  //   const formattedBiometricInfo = {
  //     ...biometricInfo,
  //     height: Number(biometricInfo.height),
  //     weight: Number(biometricInfo.weight),
  //   };

  //   const { biometricData, ...rest } = formattedBiometricInfo;
  //   const formattedForServer = {
  //     ...rest,
  //     biometric_data_by_date: biometricData,
  //   };

  //   // console.log('*분석레포트 생성 api request body:', JSON.stringify(formattedBiometricInfo));
  //   console.log('*분석레포트 생성 api request body:', JSON.stringify(formattedForServer));

  //   saveResultsToFile(JSON.stringify(formattedBiometricInfo));

  //   try {
  //     const response = await fetch('http://3.36.176.18:/prediction/report/create', {
  //       method: 'POST',
  //       headers: {
  //         // 'Content-Type': 'application/json',
  //         Authorization: `Bearer ${token}`,
  //       },
  //       body: JSON.stringify(formattedBiometricInfo),
  //     });

  //     const data = await response.json();

  //     if (!response.ok) {
  //       console.log('분석레포트 생성 실패');
  //       console.log(response);

  //       return rejectWithValue(data.message || '분석레포트 생성 실패');
  //     }

  //     console.log('분석레포트 생성 성공');
  //     console.log('*분석레포트 생성 api response:', data);

  //     return data;
  //   } catch (error) {
  //     console.log('분석레포트 생성 실패');
  //     console.log(error);

  //     return rejectWithValue(error);
  //   }
  }
);

export const fetchReport = createAsyncThunk(
  'report/fetchReport',
  async (
    { token, date }: { token: string; date: number },
    { rejectWithValue }
  ) => {
    // FOR LOCAL TEST
    // success
    console.log('레포트 조회 성공');
    // if (date === 20250609) {
      // return {
      //   'risk_score': 30,
      //   'risk_level': 'low',
      //   'question_list': [{ 'question_id': 1, 'answer': '1' }, { 'question_id': 2, 'answer': '2' }, { 'question_id': 3, 'answer': '3' }, { 'question_id': 4, 'answer': '4' }, { 'question_id': 5, 'answer': '5' }, { 'question_id': 6, 'answer': '6' }, { 'question_id': 7, 'answer': '7' }, { 'question_id': 8, 'answer': '8' }, { 'question_id': 9, 'answer': '9' }, { 'question_id': 10, 'answer': '10' }, { 'question_id': 11, 'answer': '11' }, { 'question_id': 12, 'answer': '12' }, { 'question_id': 13, 'answer': '13' }, { 'question_id': 14, 'answer': '14' }, { 'question_id': 15, 'answer': '15' }, { 'question_id': 16, 'answer': '16' }, { 'question_id': 17, 'answer': '17' }],
      //   'biometric_data_list': [{ 'biometric_data_id': 1, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 2, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 3, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 4, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 5, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 6, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 7, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 8, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 9, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 10, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 11, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 12, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 13, 'biometric_data_value': 0 }, { 'biometric_data_id': 14, 'biometric_data_value': 70620 }, { 'biometric_data_id': 15, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 16, 'biometric_data_value': 30 }, { 'biometric_data_id': 17, 'biometric_data_value': 3660 }, { 'biometric_data_id': 18, 'biometric_data_value': 18180 }, { 'biometric_data_id': 19, 'biometric_data_value': 96 }, { 'biometric_data_id': 20, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 21, 'biometric_data_value': 1 }, { 'biometric_data_id': 22, 'biometric_data_value': 9960 }, { 'biometric_data_id': 23, 'biometric_data_value': 8625 }, { 'biometric_data_id': 24, 'biometric_data_value': 15825 }, { 'biometric_data_id': 25, 'biometric_data_value': 3 }, { 'biometric_data_id': 26, 'biometric_data_value': 4560 }, { 'biometric_data_id': 27, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 28, 'biometric_data_value': 18180 }, { 'biometric_data_id': 29, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 30, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 31, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 32, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 33, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 34, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 35, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 36, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 37, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 38, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 39, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 40, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 41, 'biometric_data_value': 0 }, { 'biometric_data_id': 42, 'biometric_data_value': 70620 }, { 'biometric_data_id': 43, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 44, 'biometric_data_value': 30 }, { 'biometric_data_id': 45, 'biometric_data_value': 3660 }, { 'biometric_data_id': 46, 'biometric_data_value': 18180 }, { 'biometric_data_id': 47, 'biometric_data_value': 96 }, { 'biometric_data_id': 48, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 49, 'biometric_data_value': 1 }, { 'biometric_data_id': 50, 'biometric_data_value': 9960 }, { 'biometric_data_id': 51, 'biometric_data_value': 8625 }, { 'biometric_data_id': 52, 'biometric_data_value': 15825 }, { 'biometric_data_id': 53, 'biometric_data_value': 3 }, { 'biometric_data_id': 54, 'biometric_data_value': 4560 }, { 'biometric_data_id': 55, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 56, 'biometric_data_value': 18180 }, { 'biometric_data_id': 57, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 58, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 59, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 60, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 61, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 62, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 63, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 64, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 65, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 66, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 67, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 68, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 69, 'biometric_data_value': 0 }, { 'biometric_data_id': 70, 'biometric_data_value': 70620 }, { 'biometric_data_id': 71, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 72, 'biometric_data_value': 30 }, { 'biometric_data_id': 73, 'biometric_data_value': 3660 }, { 'biometric_data_id': 74, 'biometric_data_value': 18180 }, { 'biometric_data_id': 75, 'biometric_data_value': 96 }, { 'biometric_data_id': 76, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 77, 'biometric_data_value': 1 }, { 'biometric_data_id': 78, 'biometric_data_value': 9960 }, { 'biometric_data_id': 79, 'biometric_data_value': 8625 }, { 'biometric_data_id': 80, 'biometric_data_value': 15825 }, { 'biometric_data_id': 81, 'biometric_data_value': 3 }, { 'biometric_data_id': 82, 'biometric_data_value': 4560 }, { 'biometric_data_id': 83, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 84, 'biometric_data_value': 18180 }],
      //   'message': '레포트 조회 성공',
      // };
    // } else if (date === 20250610) {
      return {
      'risk_score': 40,
      'risk_level': 'low',
      'question_list': [{ 'question_id': 1, 'answer': '1' }, { 'question_id': 2, 'answer': '2' }, { 'question_id': 3, 'answer': '3' }, { 'question_id': 4, 'answer': '4' }, { 'question_id': 5, 'answer': '2' }, { 'question_id': 6, 'answer': '6' }, { 'question_id': 7, 'answer': '300' }, { 'question_id': 8, 'answer': '120' }, { 'question_id': 9, 'answer': '2' }, { 'question_id': 10, 'answer': '1' }, { 'question_id': 11, 'answer': '11' }, { 'question_id': 12, 'answer': '12' }, { 'question_id': 13, 'answer': '10' }, { 'question_id': 14, 'answer': '7' }, { 'question_id': 15, 'answer': '40' }, { 'question_id': 16, 'answer': '16' }, { 'question_id': 17, 'answer': '17' }],
      'biometric_data_list': [{ 'biometric_data_id': 1, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 2, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 3, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 4, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 5, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 6, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 7, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 8, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 9, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 10, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 11, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 12, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 13, 'biometric_data_value': 0 }, { 'biometric_data_id': 14, 'biometric_data_value': 70620 }, { 'biometric_data_id': 15, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 16, 'biometric_data_value': 30 }, { 'biometric_data_id': 17, 'biometric_data_value': 3660 }, { 'biometric_data_id': 18, 'biometric_data_value': 18180 }, { 'biometric_data_id': 19, 'biometric_data_value': 96 }, { 'biometric_data_id': 20, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 21, 'biometric_data_value': 1 }, { 'biometric_data_id': 22, 'biometric_data_value': 9960 }, { 'biometric_data_id': 23, 'biometric_data_value': 8625 }, { 'biometric_data_id': 24, 'biometric_data_value': 15825 }, { 'biometric_data_id': 25, 'biometric_data_value': 3 }, { 'biometric_data_id': 26, 'biometric_data_value': 4560 }, { 'biometric_data_id': 27, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 28, 'biometric_data_value': 18180 }, { 'biometric_data_id': 29, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 30, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 31, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 32, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 33, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 34, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 35, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 36, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 37, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 38, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 39, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 40, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 41, 'biometric_data_value': 0 }, { 'biometric_data_id': 42, 'biometric_data_value': 70620 }, { 'biometric_data_id': 43, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 44, 'biometric_data_value': 30 }, { 'biometric_data_id': 45, 'biometric_data_value': 3660 }, { 'biometric_data_id': 46, 'biometric_data_value': 18180 }, { 'biometric_data_id': 47, 'biometric_data_value': 96 }, { 'biometric_data_id': 48, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 49, 'biometric_data_value': 1 }, { 'biometric_data_id': 50, 'biometric_data_value': 9960 }, { 'biometric_data_id': 51, 'biometric_data_value': 8625 }, { 'biometric_data_id': 52, 'biometric_data_value': 15825 }, { 'biometric_data_id': 53, 'biometric_data_value': 3 }, { 'biometric_data_id': 54, 'biometric_data_value': 4560 }, { 'biometric_data_id': 55, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 56, 'biometric_data_value': 18180 }, { 'biometric_data_id': 57, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 58, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 59, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 60, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 61, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 62, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 63, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 64, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 65, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 66, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 67, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 68, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 69, 'biometric_data_value': 0 }, { 'biometric_data_id': 70, 'biometric_data_value': 70620 }, { 'biometric_data_id': 71, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 72, 'biometric_data_value': 30 }, { 'biometric_data_id': 73, 'biometric_data_value': 3660 }, { 'biometric_data_id': 74, 'biometric_data_value': 18180 }, { 'biometric_data_id': 75, 'biometric_data_value': 96 }, { 'biometric_data_id': 76, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 77, 'biometric_data_value': 1 }, { 'biometric_data_id': 78, 'biometric_data_value': 9960 }, { 'biometric_data_id': 79, 'biometric_data_value': 8625 }, { 'biometric_data_id': 80, 'biometric_data_value': 15825 }, { 'biometric_data_id': 81, 'biometric_data_value': 3 }, { 'biometric_data_id': 82, 'biometric_data_value': 4560 }, { 'biometric_data_id': 83, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 84, 'biometric_data_value': 18180 }],
      'message': '레포트 조회 성공',
    };
    // }
    // fail
    // console.log('레포트 조회 실패');
    // return rejectWithValue({
    //   'risk_score': 60,
    //   'risk_level': 'high',
    //   'question_list': [{ 'question_id': 1, 'answer': '1' }, { 'question_id': 2, 'answer': '2' }, { 'question_id': 3, 'answer': '3' }, { 'question_id': 4, 'answer': '4' }, { 'question_id': 5, 'answer': '5' }, { 'question_id': 6, 'answer': '6' }, { 'question_id': 7, 'answer': '7' }, { 'question_id': 8, 'answer': '8' }, { 'question_id': 9, 'answer': '9' }, { 'question_id': 10, 'answer': '10' }, { 'question_id': 11, 'answer': '11' }, { 'question_id': 12, 'answer': '12' }, { 'question_id': 13, 'answer': '13' }, { 'question_id': 14, 'answer': '14' }, { 'question_id': 15, 'answer': '15' }, { 'question_id': 16, 'answer': '16' }, { 'question_id': 17, 'answer': '17' }],
    //   'biometric_data_list': [{ 'biometric_data_id': 1, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 2, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 3, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 4, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 5, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 6, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 7, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 8, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 9, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 10, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 11, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 12, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 13, 'biometric_data_value': 0 }, { 'biometric_data_id': 14, 'biometric_data_value': 70620 }, { 'biometric_data_id': 15, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 16, 'biometric_data_value': 30 }, { 'biometric_data_id': 17, 'biometric_data_value': 3660 }, { 'biometric_data_id': 18, 'biometric_data_value': 18180 }, { 'biometric_data_id': 19, 'biometric_data_value': 96 }, { 'biometric_data_id': 20, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 21, 'biometric_data_value': 1 }, { 'biometric_data_id': 22, 'biometric_data_value': 9960 }, { 'biometric_data_id': 23, 'biometric_data_value': 8625 }, { 'biometric_data_id': 24, 'biometric_data_value': 15825 }, { 'biometric_data_id': 25, 'biometric_data_value': 3 }, { 'biometric_data_id': 26, 'biometric_data_value': 4560 }, { 'biometric_data_id': 27, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 28, 'biometric_data_value': 18180 }, { 'biometric_data_id': 29, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 30, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 31, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 32, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 33, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 34, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 35, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 36, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 37, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 38, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 39, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 40, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 41, 'biometric_data_value': 0 }, { 'biometric_data_id': 42, 'biometric_data_value': 70620 }, { 'biometric_data_id': 43, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 44, 'biometric_data_value': 30 }, { 'biometric_data_id': 45, 'biometric_data_value': 3660 }, { 'biometric_data_id': 46, 'biometric_data_value': 18180 }, { 'biometric_data_id': 47, 'biometric_data_value': 96 }, { 'biometric_data_id': 48, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 49, 'biometric_data_value': 1 }, { 'biometric_data_id': 50, 'biometric_data_value': 9960 }, { 'biometric_data_id': 51, 'biometric_data_value': 8625 }, { 'biometric_data_id': 52, 'biometric_data_value': 15825 }, { 'biometric_data_id': 53, 'biometric_data_value': 3 }, { 'biometric_data_id': 54, 'biometric_data_value': 4560 }, { 'biometric_data_id': 55, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 56, 'biometric_data_value': 18180 }, { 'biometric_data_id': 57, 'biometric_data_value': 1.5 }, { 'biometric_data_id': 58, 'biometric_data_value': 334.397 }, { 'biometric_data_id': 59, 'biometric_data_value': 2054.059 }, { 'biometric_data_id': 60, 'biometric_data_value': 53757.37 }, { 'biometric_data_id': 61, 'biometric_data_value': 16.0664 }, { 'biometric_data_id': 62, 'biometric_data_value': 782.6707333333334 }, { 'biometric_data_id': 63, 'biometric_data_value': 0.8412 }, { 'biometric_data_id': 64, 'biometric_data_value': 40.42166666666667 }, { 'biometric_data_id': 65, 'biometric_data_value': 128.5312 }, { 'biometric_data_id': 66, 'biometric_data_value': 15.653414666666668 }, { 'biometric_data_id': 67, 'biometric_data_value': 1.6824 }, { 'biometric_data_id': 68, 'biometric_data_value': 161.68666666666667 }, { 'biometric_data_id': 69, 'biometric_data_value': 0 }, { 'biometric_data_id': 70, 'biometric_data_value': 70620 }, { 'biometric_data_id': 71, 'biometric_data_value': 57.32926666666667 }, { 'biometric_data_id': 72, 'biometric_data_value': 30 }, { 'biometric_data_id': 73, 'biometric_data_value': 3660 }, { 'biometric_data_id': 74, 'biometric_data_value': 18180 }, { 'biometric_data_id': 75, 'biometric_data_value': 96 }, { 'biometric_data_id': 76, 'biometric_data_value': 82.02 }, { 'biometric_data_id': 77, 'biometric_data_value': 1 }, { 'biometric_data_id': 78, 'biometric_data_value': 9960 }, { 'biometric_data_id': 79, 'biometric_data_value': 8625 }, { 'biometric_data_id': 80, 'biometric_data_value': 15825 }, { 'biometric_data_id': 81, 'biometric_data_value': 3 }, { 'biometric_data_id': 82, 'biometric_data_value': 4560 }, { 'biometric_data_id': 83, 'biometric_data_value': 55.19200000000001 }, { 'biometric_data_id': 84, 'biometric_data_value': 18180 }],
    //   'message': '레포트 조회 실패',
    // }.message);

    // FOR SERVER COMMUNICATION
    // try {
    //   const response = await fetch(`http://3.36.176.18:/prediction/report/${date}`, {
    //     method: 'GET',
    //     headers: {
    //       Authorization: `Bearer ${token}`,
    //       // 'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({ create_date: date }),
    //   });

    //   if (!response.ok) {
    //     const errorData = await response.json();
    //     return rejectWithValue(errorData);
    //   }

    //   const data = await response.json();
    //   return data;
    // } catch (error) {
    //   return rejectWithValue(error);
    // }
  }
);

const reportSlice = createSlice({
  name: 'report',
  initialState,
  reducers: {
    resetReport(state) {
      state.loading = false;
      state.error = null;
      state.riskScore = null;
      state.riskLevel = null;
      state.questionList = [];
      state.biometricDataList = [];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(createReport.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createReport.fulfilled, (state, action: PayloadAction<any>) => {
        state.loading = false;
        state.riskScore = action.payload.risk_score;
        state.riskLevel = action.payload.risk_level;
        state.questionList = action.payload.question_list;
        state.biometricDataList = action.payload.biometric_data_list;
      })
      .addCase(createReport.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string || '레포트 생성 중 오류가 발생했습니다.';
      })
      .addCase(fetchReport.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchReport.fulfilled, (state, action: PayloadAction<any>) => {
        state.loading = false;
        state.riskScore = action.payload.risk_score;
        state.riskLevel = action.payload.risk_level;
        state.questionList = action.payload.question_list;
        state.biometricDataList = action.payload.biometric_data_list;
      })
      .addCase(fetchReport.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});

export const { resetReport } = reportSlice.actions;
export default reportSlice.reducer;