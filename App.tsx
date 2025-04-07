import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import SplashScreen from './src/screens/SplashScreen.tsx';
import HomeScreen from './src/screens/main/HomeScreen.tsx';
import LoginScreen from './src/screens/auth/LoginScreen.tsx';
import EmailSignUpScreen from './src/screens/auth/EmailSignUpScreen.tsx';
import LifestyleStartScreen from './src/screens/lifestyle/LifestyleStartScreen.tsx';
import LifestyleQuestionScreen from './src/screens/lifestyle/LifestyleQuestionScreen.tsx';
import LifestyleCompleteScreen from './src/screens/lifestyle/LifestyleCompleteScreen.tsx';

const Stack = createStackNavigator();

function App(): React.JSX.Element {
  return (
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
          name="EmailSignUp"
          component={EmailSignUpScreen}
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
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;