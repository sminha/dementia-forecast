import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface SignupState {
  formData: {
    email: string;
    password: string;
    name: string;
    gender: string;
    birthdate: string;
    phone: string;
    address: string;
    detailAddress: string;
    agreeAll: boolean;
    privacyShare: boolean;
    privacyUse: boolean;
  };
}

const initialState: SignupState = {
  formData: {
    email: '',
    password: '',
    name: '',
    gender: '',
    birthdate: '',
    phone: '',
    address: '',
    detailAddress: '',
    agreeAll: false,
    privacyShare: false,
    privacyUse: false,
  },
};

const signupSlice = createSlice({
  name: 'signup',
  initialState,
  reducers: {
    setFormData: (state, action: PayloadAction<{field: string, value: string | boolean}>) => {
      (state.formData as any)[action.payload.field] = action.payload.value;
    },
  },
});

export const { setFormData } = signupSlice.actions;
export default signupSlice.reducer;