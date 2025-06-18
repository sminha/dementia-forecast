export type RootStackParamList = {
  Splash: undefined;
  Home: undefined;
  Login: undefined;
  EmailLogin: undefined;
  EmailSignUp: undefined;
  PrivacyUsePolicy: undefined;
  PrivacySharePolicy: undefined;
  EmailSignUpComplete: undefined;
  LifestyleStart: undefined;
  LifestyleQuestion: undefined;
  LifestyleComplete: undefined;
  BiometricStart: undefined;
  BiometricInput: undefined;
  BiometricOverview: { from: string };
  BiometricFetchComplete: undefined;
  BiometricSubmitComplete: undefined;
  ReportStart: undefined;
  ReportResult:
    | { type: 'reportStart'; response: string }
    | { type: 'reportView'; data: any }
    | { type: 'Home'; data: any };
    // | { type: 'more'; data: any };
  Mypage: undefined;
  AccountView: undefined;
  PasswordEdit: undefined;
  NameEdit: undefined;
  GenderEdit: undefined;
  BirthdateEdit: undefined;
  PhoneEdit: undefined;
  AddressEdit: undefined;
  LifestyleView: undefined;
  LifestyleEdit: { topic: string };
  ReportView: { from : string };
  AccountDelete: undefined;
  DementiaInfoList: undefined;
  DementiaInfoDetail: undefined;
};