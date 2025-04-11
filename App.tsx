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
import BiometricCompleteScreen from './src/screens/biometric/BiometricCompleteScreen.tsx';

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
            name="BiometricComplete"
            component={BiometricCompleteScreen}
            options={{ headerShown: false }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </Provider>
  );
}

export default App;