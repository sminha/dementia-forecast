import { createSlice, PayloadAction, createAsyncThunk } from '@reduxjs/toolkit';

interface LoginState {
  formData: {
    email: string;
    password: string;
  };
  isLoading: boolean;
  error: string | null;
  loginResult: {
    statusCode?: number;
    message?: string;
    accessToken?: string | null;
    refreshToken?: string | null;
    name?: string;
  } | null;
  tokens: {
    accessToken: string | null;
    refreshToken: string | null;
  };
}

const initialState: LoginState = {
  formData: {
    email: '',
    password: '',
  },
  isLoading: false,
  error: null,
  loginResult: null,
  tokens: {
    accessToken: null,
    refreshToken: null,
  },
};

export const loginUser = createAsyncThunk(
  'login/loginUser',
  async (formData: { email: string; password: string }, { rejectWithValue }) => {
    // FOR LOCAL TEST
    if (formData.email === 'test@example.com') {
      return { statusCode: 200, message: '로그인 성공', accessToken: 'a', refreshToken: 'r', name: '홍길동' };
    } else {
      return rejectWithValue({ statusCode: 400, message: '로그인 실패', accessToken: null, refreshToken: null, name: '홍길동' }.message);
    }

    // FOR SERVER COMMUNIATION
    // try {
    //   const response = await fetch('BACKEND_URL/auth/login', {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({
    //       email: formData.email,
    //       password: formData.password,
    //     }),
    //   });

    //   const data = await response.json();

    //   if (data.statusCode !== 200) {
    //     return rejectWithValue(data.message);
    //   }

    //   // return {
    //   //   accessToken: data.accessToken,
    //   //   refreshToken: data.refreshToken,
    //   // };

    //   return data;
    // } catch (error: any) {
    //   return rejectWithValue(error?.message || '알 수 없는 오류가 발생했습니다.');
    // }
  }
);

const loginSlice = createSlice({
  name: 'login',
  initialState,
  reducers: {
    setLoginFormData: (state, action: PayloadAction<{field: string; value: string}>) => {
      (state.formData as any)[action.payload.field] = action.payload.value;
    },
    clearLoginResult: (state) => {
      state.isLoading = false;
      state.error = null;
      state.loginResult = null;
      state.tokens = { accessToken: null, refreshToken: null };
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(loginUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.loginResult = null;
        state.tokens = { accessToken: null, refreshToken: null };
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.error = null;
        state.loginResult = action.payload;
        state.tokens.accessToken = action.payload.accessToken;
        state.tokens.refreshToken = action.payload.refreshToken;
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.loginResult = { message: action.payload as string };
        state.tokens = { accessToken: null, refreshToken: null };
      });
  },
});

export const { setLoginFormData, clearLoginResult } = loginSlice.actions;
export default loginSlice.reducer;