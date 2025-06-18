import { configureStore } from '@reduxjs/toolkit';
import signupReducer from './slices/signupSlice.ts';
import loginReducer from './slices/loginSlice.ts';
import userReducer from './slices/userSlice.ts';
import lifestyleReducer from './slices/lifestyleSlice.ts';
import biometricReducer from './slices/biometricSlice.ts';
import reportReducer from './slices/reportSlice.ts';

export const store = configureStore({
  reducer: {
    signup: signupReducer,
    login: loginReducer,
    user: userReducer,
    lifestyle: lifestyleReducer,
    biometric: biometricReducer,
    report: reportReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
      immutableCheck: false,
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;