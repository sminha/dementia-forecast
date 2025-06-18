import React from 'react';
import { Provider } from 'react-redux';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { store } from './src/redux/store.ts';
import SplashScreen from './src/screens/SplashScreen.tsx';
import HomeScreen from './src/screens/main/HomeScreen.tsx';
import LoginScreen from './src/screens/auth/LoginScreen.tsx';
import EmailLoginScreen from './src/screens/auth/EmailLoginScreen.tsx';
import EmailSignUpScreen from './src/screens/auth/EmailSignUpScreen.tsx';
import PrivacyUsePolicyScreen from './src/screens/policy/PrivacyUsePolicyScreen.tsx';
import PrivacySharePolicyScreen from './src/screens/policy/PrivacySharePolicyScreen.tsx';
import EmailSignUpCompleteScreen from './src/screens/auth/EmailSignUpCompleteScreen.tsx';
import LifestyleStartScreen from './src/screens/lifestyle/LifestyleStartScreen.tsx';
import LifestyleQuestionScreen from './src/screens/lifestyle/LifestyleQuestionScreen.tsx';
import LifestyleCompleteScreen from './src/screens/lifestyle/LifestyleCompleteScreen.tsx';
import BiometricStartScreen from './src/screens/biometric/BiometricStartScreen.tsx';
import BiometricInputScreen from './src/screens/biometric/BiometricInputScreen.tsx';
import BiometricFetchCompleteScreen from './src/screens/biometric/BiometricFetchCompleteScreen.tsx';
import BiometricSubmitCompleteScreen from './src/screens/biometric/BiometricSubmitCompleteScreen.tsx';
import BiometricOverviewScreen from './src/screens/biometric/BiometricOverviewScreen.tsx';
import ReportStartScreen from './src/screens/report/ReportStartScreen.tsx';
import ReportResultScreen from './src/screens/report/ReportResultScreen.tsx';
import MypageScreen from './src/screens/mypage/MypageScreen.tsx';
import AccountViewScreen from './src/screens/mypage/view/AccountViewScreen.tsx';
import PasswordEditScreen from './src/screens/mypage/edit/PasswordEditScreen.tsx';
import NameEditScreen from './src/screens/mypage/edit/NameEditScreen.tsx';
import GenderEditScreen from './src/screens/mypage/edit/GenderEditScreen.tsx';
import BirthdateEditScreen from './src/screens/mypage/edit/BirthdateEditScreen.tsx';
import PhoneEditScreen from './src/screens/mypage/edit/PhoneEditScreen.tsx';
import AddressEditScreen from './src/screens/mypage/edit/AddressEditScreen.tsx';
import ReportViewScreen from './src/screens/mypage/view/ReportViewScreen.tsx';
import LifestyleViewScreen from './src/screens/mypage/view/LifestyleViewScreen.tsx';
import LifestyleEditScreen from './src/screens/mypage/edit/LifestyleEditScreen.tsx';
import AccountDeleteScreen from './src/screens/mypage/AccountDeleteScreen.tsx';
import DementiaInfoListScreen from './src/screens/dementiaInfo/DementiaInfoListScreen.tsx';
import DementiaInfoDetailScreen from './src/screens/dementiaInfo/DementiaInfoDetailScreen.tsx';

const Stack = createStackNavigator();

function App(): React.JSX.Element {
  return (
    <Provider store={store}>
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Splash">
          <Stack.Screen
            name="Splash"
            component={SplashScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="Home"
            component={HomeScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="Login"
            component={LoginScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="EmailLogin"
            component={EmailLoginScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="EmailSignUp"
            component={EmailSignUpScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="PrivacyUsePolicy"
            component={PrivacyUsePolicyScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="PrivacySharePolicy"
            component={PrivacySharePolicyScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="EmailSignUpComplete"
            component={EmailSignUpCompleteScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="LifestyleStart"
            component={LifestyleStartScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="LifestyleQuestion"
            component={LifestyleQuestionScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="LifestyleComplete"
            component={LifestyleCompleteScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="BiometricStart"
            component={BiometricStartScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="BiometricInput"
            component={BiometricInputScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="BiometricFetchComplete"
            component={BiometricFetchCompleteScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="BiometricSubmitComplete"
            component={BiometricSubmitCompleteScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="BiometricOverview"
            component={BiometricOverviewScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="ReportStart"
            component={ReportStartScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="ReportResult"
            component={ReportResultScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="Mypage"
            component={MypageScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="AccountView"
            component={AccountViewScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="PasswordEdit"
            component={PasswordEditScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="NameEdit"
            component={NameEditScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="GenderEdit"
            component={GenderEditScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="BirthdateEdit"
            component={BirthdateEditScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="PhoneEdit"
            component={PhoneEditScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="AddressEdit"
            component={AddressEditScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="ReportView"
            component={ReportViewScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="LifestyleView"
            component={LifestyleViewScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="LifestyleEdit"
            component={LifestyleEditScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="AccountDelete"
            component={AccountDeleteScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="DementiaInfoList"
            component={DementiaInfoListScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="DementiaInfoDetail"
            component={DementiaInfoDetailScreen}
            options={{ headerShown: false }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </Provider>
  );
}

export default App;