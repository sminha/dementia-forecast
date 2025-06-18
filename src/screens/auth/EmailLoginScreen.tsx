import React, { useState, useEffect } from 'react';
import { View, SafeAreaView, TouchableOpacity, ScrollView, TextInput, StyleSheet } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { setLoginFormData, clearLoginResult, loginUser } from '../../redux/slices/loginSlice.ts';
import { fetchUser } from '../../redux/slices/userSlice.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import { storeTokens, loadTokens } from '../../redux/actions/authAction.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const EmailLoginScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'EmailLogin'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const formData = useSelector((state: RootState) => state.login.formData);
  const loginResult = useSelector((state: RootState) => state.login.loginResult);
  const isLoading = useSelector((state: RootState) => state.login.isLoading);

  const userInfo = useSelector((state: RootState) => state.user.userInfo); // 로그아웃 또는 탈퇴 후 사용자 정보 삭제 확인 위함

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [secureText, setSecureText] = useState(true);

  useEffect(() => {
    // console.log(userInfo); // 로그아웃 또는 탈퇴 후 사용자 정보 삭제 확인 위함
    // console.log(loginResult);

    const finalizeLogin = async () => {
      // if (loginResult?.statusCode === 200) {
      if (loginResult?.message === '로그인 성공') {
        if (loginResult.accessToken && loginResult.refreshToken) {
          await storeTokens(loginResult.accessToken, loginResult.refreshToken);

          const handleFetch = async () => {
            const { accessToken } = await loadTokens();
            if (!accessToken) {
              console.log('로그인 정보가 없습니다.');
              return;
            }

            dispatch(fetchUser(accessToken));
          };
          handleFetch();

          // console.log(userInfo);
          // console.log('사용자 정보:', JSON.stringify(userInfo));

          // await AsyncStorage.setItem('loggedInUser', JSON.stringify(userInfo));
          navigation.replace('Home');
        }
      } else if (loginResult && loginResult.message !== '로그인 성공') {
        dispatch(clearLoginResult());
      }
    };

    finalizeLogin();
  }, [loginResult, navigation, dispatch, userInfo]);

  // useEffect(() => {
  //   if (userInfo) {
  //     AsyncStorage.setItem('loggedInUser', JSON.stringify(userInfo));
  //   }
  // }, [userInfo]);

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isFormValid = () => {
    return email.trim() !== '' && password.trim() !== '';
  };

  const handleLogin = () => {
    dispatch(loginUser(formData));
  };

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
      </SafeAreaView>

      <View style={styles.headerContainer}>
        <CustomText style={styles.titleText}>이메일로 로그인하기</CustomText>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContentContainer} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'email' && styles.labelTextFocused]}>이메일</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'email' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('email')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => {
                setEmail(text);
                dispatch(setLoginFormData({ field: 'email', value: text }));
              }}
              value={email}
            />
            {email !== '' &&
            <TouchableOpacity onPress={() => setEmail('')} style={styles.clearButton}>
              <Icon name="close-circle" size={20} color="#B4B4B4" />
            </TouchableOpacity>
            }
          </View>
        </View>

        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'password' && styles.labelTextFocused]}>비밀번호</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'password' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('password')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => {
                setPassword(text);
                dispatch(setLoginFormData({ field: 'password', value: text }));
              }}
              secureTextEntry={secureText}
              value={password}
            />
            {password !== '' && (
              <TouchableOpacity onPress={() => setSecureText(!secureText)} style={styles.clearButton}>
                <Icon name={secureText ? 'eye' : 'eye-off'} size={20} color="#B4B4B4" />
              </TouchableOpacity>
            )}
          </View>
        </View>
      </ScrollView>

      {loginResult && loginResult.message !== '로그인 성공' &&
        <CustomText style={styles.failText}>로그인에 실패하였습니다. 아이디와 비밀번호를 정확히 입력해주세요.</CustomText>
      }

      <View>
        <TouchableOpacity
          onPress={handleLogin}
          style={[styles.actionButton, isFormValid() ? styles.actionButtonEnabled : styles.actionButtonDisabled]}
          // disabled={!isFormValid()}>
          disabled={!isFormValid() || isLoading}>

          {/* <CustomText style={[styles.actionButtonText, isFormValid() ? styles.actionButtonTextEnabled : styles.actionButtonTextDisabled]}>
            로그인
          </CustomText> */}
          <CustomText style={[styles.actionButtonText, isFormValid() && !isLoading ? styles.actionButtonTextEnabled : styles.actionButtonTextDisabled]}>
            {isLoading ? '로그인 중...' : '로그인'}
          </CustomText>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    backgroundColor: '#FFFFFF',
  },
  safeAreaWrapper: {
    width: 70,
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  headerContainer: {
    paddingTop: 40,
    paddingBottom: 50,
    paddingHorizontal: 20,
  },
  backButton: {
    marginBottom: 10,
  },
  titleText: {
    fontSize: 24,
  },
  scrollContentContainer: {
    paddingHorizontal: 20,
  },
  inputGroup: {
    marginBottom: 40,
  },
  labelText: {
    fontSize: 18,
    color: '#202020',
  },
  labelTextFocused: {
    color: '#9F8473',
  },
  inputFieldWrapper: {
    position: 'relative',
    flexDirection: 'row',
    alignItems: 'center',
  },
  clearButton: {
    position: 'absolute',
    top: '50%',
    right: 1,
    transform: [{ translateY: -8 }],
  },
  inputField: {
    width: '100%',
    paddingRight: 35,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  inputFieldFocused: {
    borderBottomColor: '#9F8473',
  },
  chevronIcon: {
    marginLeft: 'auto',
  },
  failText: {
    color: '#CA4747',
  },
  actionButton: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    borderRadius: 5,
  },
  actionButtonEnabled: {
    backgroundColor: '#D6CCC2',
  },
  actionButtonDisabled: {
    backgroundColor: '#F2EFED',
  },
  actionButtonText: {
    fontSize: 20,
  },
  actionButtonTextEnabled: {
    color: '#575553',
  },
  actionButtonTextDisabled: {
    color: '#B4B4B4',
  },
});

export default EmailLoginScreen;