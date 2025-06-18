import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';

export type HouseholdType = '일반 가정' | '결손 가정' | '나홀로 가정' | '저소득 가정' | '저소득 결손 가정' | '저소득 나홀로 가정';

export interface LifestyleInfo {
  householdSize: number;
  hasSpouse: boolean;
  income: number;
  expenses: number;
  alcoholExpense: number;
  tobaccoExpense: number;
  bookExpense: number;
  welfareExpense: number;
  medicalExpense: number;
  insuranceExpense: number;
  hasPet: boolean;
  householdType: HouseholdType;
}

interface LifestyleState {
  lifestyleInfo: LifestyleInfo;
  isLoading: boolean;
  error: string | null;
  saveResult: {
    statusCode?: number;
    message: string;
  } | null;
  fetchResult: {
    statusCode?: number;
    message: string;
  } | null;
}

const initialState: LifestyleState = {
  lifestyleInfo: {
    householdSize: 0,
    hasSpouse: false,
    income: 0,
    expenses: 0,
    alcoholExpense: 0,
    tobaccoExpense: 0,
    bookExpense: 0,
    welfareExpense: 0,
    medicalExpense: 0,
    insuranceExpense: 0,
    hasPet: false,
    householdType: '일반 가정' as HouseholdType,
  },
  isLoading: false,
  error: null,
  saveResult: null,
  fetchResult: null,
};

export const saveLifestyle = createAsyncThunk(
  '/lifestyle/save',
  async (
    { token, questionList }: { token: string, questionList: { question_id: number; answer: string }[]}, { rejectWithValue }
  ) => {
    // FOR LOCAL TEST
    // if (questionList[0].answer === '1') {
    //   return { statusCode: 200, message: '라이프스타일 저장 성공' };
    // } else if (questionList[0].answer === '2') {
    //   return rejectWithValue('라이프스타일 저장 실패');
    // } else {
    //   return rejectWithValue('인증 실패');
    // }

    // FOR SERVER COMMUNICATION
    console.log('*라이프스타일 저장 api request body:', { question_list: questionList });

    try {
      const response = await fetch('https://d1kt6v32r7kma5.cloudfront.net/lifestyle/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ question_list: questionList }),
      });

      const data = await response.json();

      if (!response.ok) {
        console.log('라이프스타일 저장 실패');
        console.log(response);

        return rejectWithValue(data.message || '라이프스타일 저장 실패');
      }

      console.log('라이프스타일 저장 성공');
      console.log('*라이프스타일 저장 api response:', data);

      // return {
      //   statusCode: response.data.statusCode,
      //   message: response.data.message,
      // };
      return data;
    } catch (error: any) {
      console.log('라이프스타일 저장 실패');
      console.log(error);

      return rejectWithValue('알 수 없는 오류가 발생했습니다.');
    }
  },
);

export const fetchLifestyle = createAsyncThunk(
  'lifestyle/fetch',
  async(token: string, { rejectWithValue }) => {
    // FOR LOCAL TEST
    // success
    // console.log('라이프스타일 조회 성공');
    // return {
    //   lifestyleInfo: {
    //       householdSize: 2,
    //       hasSpouse: true,
    //       income: 70,
    //       expenses: 30,
    //       alcoholExpense: 40,
    //       tobaccoExpense: 40,
    //       bookExpense: 8,
    //       welfareExpense: 3,
    //       medicalExpense: 2,
    //       insuranceExpense: 9,
    //       hasPet: false,
    //       householdType: '일반 가정' as HouseholdType,
    //     },
    //     fetchResult: {
    //       statusCode: 200,
    //       message: '라이프스타일 조회 성공',
    //     },
    // };
    // fail
    // console.log('라이프스타일 조회 실패');
    // return rejectWithValue({
    //   statusCode: 400,
    //   message: '라이프스타일 조회 실패',
    //   question_list: [],
    // }.message);

    // FOR SERVER COMMUNICATION
    try {
      const response = await fetch('https://d1kt6v32r7kma5.cloudfront.net/lifestyle/send', {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        console.log('라이프스타일 조회 실패');
        console.log(response);

        return rejectWithValue(data.message || '라이프스타일 조회 실패');
      }

      const getAnswer = (id: number): string => data.question_list.find((q: { question_id: number; answer: string }) => q.question_id === id).answer;

      console.log('라이프스타일 조회 성공');
      console.log('*라이프스타일 조회 api response:', data);

      // return data;
      return {
        lifestyleInfo: {
          // householdSize: parseInt(getAnswer(1), 10),
          // hasSpouse: getAnswer(2) === '있음',
          // income: parseInt(getAnswer(3), 10),
          // expenses: parseInt(getAnswer(4), 10),
          // alcoholExpense: parseInt(getAnswer(5), 10),
          // tobaccoExpense: parseInt(getAnswer(6), 10),
          // bookExpense: parseInt(getAnswer(7), 10),
          // welfareExpense: parseInt(getAnswer(8), 10),
          // medicalExpense: parseInt(getAnswer(9), 10),
          // insuranceExpense: parseInt(getAnswer(10), 10),
          // hasPet: getAnswer(11) === '있음',
          // householdType: getAnswer(12) as HouseholdType,
          householdSize: parseInt(getAnswer(5), 10),
          hasSpouse: getAnswer(6) === '있음',
          income: parseInt(getAnswer(7), 10),
          expenses: parseInt(getAnswer(8), 10),
          alcoholExpense: parseInt(getAnswer(9), 10),
          tobaccoExpense: parseInt(getAnswer(10), 10),
          bookExpense: parseInt(getAnswer(11), 10),
          welfareExpense: parseInt(getAnswer(12), 10),
          medicalExpense: parseInt(getAnswer(13), 10),
          insuranceExpense: parseInt(getAnswer(14), 10),
          hasPet: getAnswer(15) === '있음',
          householdType: getAnswer(16) as HouseholdType,
        },
        fetchResult: {
          statusCode: data.statusCode,
          message: data.message,
        },
      };
    } catch (error) {
      console.log('라이프스타일 조회 실패');
      console.log(error);

      return rejectWithValue('알 수 없는 오류가 발생했습니다.');
    }
  }
);

const lifestyleSlice = createSlice({
  name: 'lifestyle',
  initialState,
  reducers: {
    setLifestyleInfo: (state, action: PayloadAction<{ field: keyof LifestyleInfo, value: number | boolean | HouseholdType }>) => {
      (state.lifestyleInfo[action.payload.field] as typeof action.payload.value) = action.payload.value;
    },
    clearSaveResult: (state) => {
      state.error = null;
      state.fetchResult = null;
    },
    clearFetchResult: (state) => {
      state.error = null;
      state.fetchResult = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(saveLifestyle.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.saveResult = null;
      })
      .addCase(saveLifestyle.fulfilled, (state, action) => {
        state.isLoading = false;
        // state.saveResult = {
        //   statusCode: action.payload.statusCode,
        //   message: action.payload.message,
        // };
        state.saveResult = action.payload.saveResult;
      })
      .addCase(saveLifestyle.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.saveResult = { message: action.payload as string };
      })
      .addCase(fetchLifestyle.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.fetchResult = null;
      })
      .addCase(fetchLifestyle.fulfilled, (state, action) => {
        state.isLoading = false;
        state.lifestyleInfo = action.payload.lifestyleInfo;
        state.fetchResult = action.payload.fetchResult;
      })
      .addCase(fetchLifestyle.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.fetchResult = { message: action.payload as string };
      });
  },
});

export const { setLifestyleInfo, clearSaveResult,  clearFetchResult } = lifestyleSlice.actions;
export default lifestyleSlice.reducer;