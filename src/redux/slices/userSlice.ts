import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface UserInfo {
  email: string;
  password: string;
  name: string;
  birthdate: string;
  gender: string;
  phone: string;
  address: string;
}

interface UserState {
  userInfo: UserInfo;
  isLoading: boolean;
  error: string | null;
  updateResult: {
    statusCode?: number;
    message: string;
  } | null;
  fetchResult: {
    statusCode?: number;
    message: string;
  } | null;
  logoutResult: {
    statusCode?: number,
    message: string,
  } | null;
  deleteResult :{
    statusCode?: number,
    message: string,
  } | null;
}

const initialState: UserState = {
  userInfo: {
    email: '',
    password: '',
    name: '',
    birthdate: '',
    gender: '',
    phone: '',
    address: '',
  },
  isLoading: false,
  error: null,
  updateResult: null,
  fetchResult: null,
  logoutResult: null,
  deleteResult: null,
};

function isoStringToNumber(dateString: string): number {
  const date = new Date(dateString);
  const year = date.getUTCFullYear();
  const month = (date.getUTCMonth() + 1).toString().padStart(2, '0'); // 월은 0부터 시작하니까 +1
  const day = date.getUTCDate().toString().padStart(2, '0');

  return Number(`${year}${month}${day}`);
}

export const updateUser = createAsyncThunk(
  'user/update',
  async (
    { token, userInfo }: { token: string, userInfo: UserInfo }, { rejectWithValue }
  ) => {
    // FOR LOCAL TEST
    // success
    // console.log('계정 정보 수정 성공');
    // return { statusCode: 200, message: '계정 정보 수정 성공' };
    // fail
    // console.log('계정 정보 수정 실패');
    // return rejectWithValue({ statusCode: 400, message: '계정 정보 수정 실패' }.message);

    // FOR SERVER COMMUNIATION
    try {
      console.log('*회원 정보 수정 api request body:', JSON.stringify({
                                                        password: userInfo.password,
                                                        name: userInfo.name,
                                                        // dob: parseInt(userInfo.birthdate, 10),
                                                        dob: isoStringToNumber(userInfo.birthdate),
                                                        gender: userInfo.gender,
                                                        contact: userInfo.phone,
                                                        // address: `${userInfo.address} ${userInfo.detailAddress}`,
                                                        address: userInfo.address,
                                                        consent: 1,
                                                      })
      );

      const response = await fetch('https://d1kt6v32r7kma5.cloudfront.net/user/update', {
        // method: 'POST',
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          password: userInfo.password,
          name: userInfo.name,
          // dob: parseInt(userInfo.birthdate, 10),
          dob: isoStringToNumber(userInfo.birthdate),
          gender: userInfo.gender,
          contact: userInfo.phone,
          // address: `${userInfo.address} ${userInfo.detailAddress}`,
          address: userInfo.address,
          consent: 1,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        return rejectWithValue(data.message || '회원 정보 수정 실패');
      }

      console.log('회원 정보 수정 성공');
      console.log('*회원 정보 수정 api response:', data);

      return data;
    } catch (error: unknown) {
      if (error instanceof Error) {
        console.log('회원 정보 수정 실패');
        console.log(error);

        return rejectWithValue(error.message);
      } else {
        console.log('회원 정보 수정 실패');
        console.log(error);

        return rejectWithValue('알 수 없는 오류가 발생했습니다.');
      }
    }
  }
);

export const fetchUser = createAsyncThunk(
  'user/fetch',
  async (token: string, { rejectWithValue }) => {
    // FOR LOCAL TEST
    // success
    // console.log('회원 정보 조회 성공');
    // return {
    //   userInfo: {
    //     password: '',
    //     email: 'honggildong@example.com',
    //     name: '홍길동',
    //     birthdate: (601111).toString().padStart(6, '0'),
    //     gender: '남',
    //     phone: '01012345678',
    //     address: '서울 중구 필동로1길 30 101호',
    //   },
    //   fetchResult: {
    //     statusCode: 200,
    //     message: '회원 정보 조회 성공',
    //   },
    // };
    // fail
    // console.log('회원 정보 조회 실패');
    // return rejectWithValue({
    //   statusCode: 400,
    //   message: '회원 정보 조회 실패',
    //   email: 'honggildong@example.com',
    //   name: '',
    //   dob: '',
    //   gender: '',
    //   contact: '',
    //   address: '',
    // }.message);

    // FOR SERVER COMMUNIATION
    try {
      const response = await fetch('https://d1kt6v32r7kma5.cloudfront.net/user/profile', {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        return rejectWithValue(data.message || '마이페이지 실패');
      }

      console.log('마이페이지 성공');
      console.log('*마이페이지 api response:', data);

      return {
        // userInfo: {
        //   password: '',
        //   email: data.email,
        //   name: data.name,
        //   birthdate: data.dob.toString().padStart(6, '0'),
        //   gender: data.gender,
        //   phone: data.contact,
        //   address: data.address,
        //   detailAddress: '',
        // },
        // fetchResult: {
        //   statusCode: data.statusCode,
        //   message: data.message,
        // },
        userInfo: {
          password: '',
          email: data.user.email,
          name: data.user.name,
          birthdate: data.user.dob.toString().padStart(6, '0'),
          gender: data.user.gender,
          phone: data.user.contact,
          address: data.user.address,
          detailAddress: '',
        },
        fetchResult: {
          statusCode: data.statusCode,
          message: data.message,
        },
      };
    } catch (error) {
      console.log('마이페이지 실패');
      console.log(error);

      return rejectWithValue('알 수 없는 오류가 발생했습니다.');
    }
  }
);

export const logoutUser = createAsyncThunk(
  'user/logout',
  async (token: string, { rejectWithValue }) => {
    // FOR TEST
    // success
    console.log('로그아웃 성공');
    return { statusCode: 200, message: '로그아웃 성공' };
    // fail
    // console.log('로그아웃 실패');
    // return rejectWithValue({ statusCode: 400, message: '로그아웃 실패' }.message);

    // FOR SERVER COMMUNICATION
    // try {
    //   const response = await fetch('BACKEND/user/logout', {
    //     method: 'POST',
    //     headers: {
    //       Authorization: `Bearer ${token}`,
    //     },
    //   });

    //   const data = await response.json();

    //   if (!response.ok) {
    //     return rejectWithValue(data.message || '로그아웃 실패');
    //   }

    //   return data;
    // } catch (error) {
    //   return rejectWithValue('알 수 없는 오류가 발생했습니다.');
    // }
  }
);

export const deleteUser = createAsyncThunk(
  'user/delete',
  async (token: string, { rejectWithValue }) => {
    // FOR TEST
    // success
    console.log('회원 탈퇴 성공');
    return { statusCode: 200, message: '회원 탈퇴 성공' };
    // fail
    // console.log('회원 탈퇴 실패');
    // return rejectWithValue({ statusCode: 400, message: '회원 탈퇴 실패' }.message);

    // FOR SERVER COMMUNICATION
    // try {
    //   const response = await fetch('BACKEND_URL/user/delete', {
    //     method: 'DELETE',
    //     headers: {
    //       Authorization: `Bearer ${token}`,
    //     },
    //   });

    //   const data = await response.json();

    //   if (!response.ok) {
    //     return rejectWithValue(data.message || '회원 탈퇴 실패');
    //   }

    //   return data;
    // } catch (error) {
    //   return rejectWithValue('알 수 없는 오류가 발생했습니다.');
    // }
  }
);

// /lifestyle/delete
export const deleteLifestyleData = createAsyncThunk(
  'user/deleteLifestyleData',
  async (token: string, { rejectWithValue }) => {
    try {
      const response = await fetch('BACKEND_URL/lifestyle/delete', {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        return rejectWithValue(data.message || '라이프스타일 데이터 삭제 실패');
      }

      return data;
    } catch (error) {
      return rejectWithValue('라이프스타일 삭제 중 오류 발생');
    }
  }
);

// /prediction/report/delete
export const deletePredictionReport = createAsyncThunk(
  'user/deletePredictionReport',
  async (token: string, { rejectWithValue }) => {
    try {
      const response = await fetch('BACKEND_URL/prediction/report/delete', {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        return rejectWithValue(data.message || '레포트 삭제 실패');
      }

      return data;
    } catch (error) {
      return rejectWithValue('레포트 삭제 중 오류 발생');
    }
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setUserInfo: (state, action: PayloadAction<{ field: keyof UserInfo, value: string }>) => {
      state.userInfo[action.payload.field] = action.payload.value;
    },
    logout: (state) => {
      state.userInfo = {
        email: '',
        password: '',
        name: '',
        birthdate: '',
        gender: '',
        phone: '',
        address: '',
      };
      state.isLoading = false;
      state.error = null;
      state.updateResult = null;
      state.fetchResult = null;
      state.deleteResult = null;
    },
    deleteAction: (state) => {
      state.userInfo = {
        email: '',
        password: '',
        name: '',
        birthdate: '',
        gender: '',
        phone: '',
        address: '',
      };
      state.isLoading = false;
      state.error = null;
      state.updateResult = null;
      state.fetchResult = null;
      state.deleteResult = null;
    },
    clearUpdateResult: (state) => {
      state.error = null;
      state.updateResult = null;
    },
    clearFetchResult: (state) => {
      state.error = null;
      state.fetchResult = null;
    },
    clearLogoutResult: (state) => {
      state.error = null;
      state.logoutResult = null;
    },
    clearDeleteResult: (state) => {
      state.error = null;
      state.deleteResult = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(updateUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.updateResult = null;
      })
      .addCase(updateUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.updateResult = action.payload;
      })
      .addCase(updateUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.updateResult = { message: action.payload as string };
      })
      .addCase(fetchUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.updateResult = null;
      })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.userInfo = action.payload.userInfo;
        state.fetchResult = action.payload.fetchResult;
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.fetchResult = { message: action.payload as string };
      })
      .addCase(logoutUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.logoutResult = null;
      })
      .addCase(logoutUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.logoutResult = action.payload;
      })
      .addCase(logoutUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.logoutResult = { message: action.payload as string };
      })
      .addCase(deleteUser.pending, (state) => {
        state.isLoading = true;
        state.error = null;
        state.deleteResult = null;
      })
      .addCase(deleteUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.deleteResult = action.payload;
      })
      .addCase(deleteUser.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.deleteResult = { message: action.payload as string };
      });
  },
});

export const { setUserInfo, logout, deleteAction, clearUpdateResult, clearFetchResult, clearLogoutResult, clearDeleteResult } = userSlice.actions;
export default userSlice.reducer;