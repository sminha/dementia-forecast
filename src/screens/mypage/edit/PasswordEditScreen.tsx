import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, TextInput, StyleSheet } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../../types/navigationTypes.ts';
import { setUserInfo, updateUser, clearUpdateResult } from '../../../redux/slices/userSlice.ts';
import { AppDispatch, RootState } from '../../../redux/store.ts';
import { loadTokens } from '../../../redux/actions/authAction.ts';
import CustomText from '../../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const PasswordEditScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const updateResult = useSelector((state: RootState) => state.user.updateResult);

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [securePassword, setSecurePassword] = useState(true);
  const [secureConfirmPassword, setSecureConfirmPassword] = useState(true);

  useEffect(() => {
    // if (updateResult?.statusCode === 200) {
    if (updateResult?.message === '회원 정보가 수정되었습니다.') {
      dispatch(setUserInfo({ field: 'password', value: password }));
      dispatch(clearUpdateResult());
      navigation.goBack();
    } else if (updateResult && updateResult.message !== '회원 정보가 수정되었습니다.') {
      dispatch(clearUpdateResult());
    }
  }, [updateResult, password, navigation, dispatch]);

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isFormValid = () => {
    const lengthValid = password.length >= 8;
    const hasUpper = /[A-Z]/.test(password);
    const hasLower = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[^A-Za-z0-9]/.test(password);

    const conditionsMet = [hasUpper, hasLower, hasNumber, hasSpecial].filter(Boolean).length >= 4;

    return lengthValid && conditionsMet;
  };

  const handleUpdate = async () => {
    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    // dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo }, updatedFields: { password } }));
    dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo, password } }));
    // dispatch(updateUser({ token: accessToken, userInfo }));
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => navigation.goBack()}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <TouchableOpacity style={styles.titleRow}>
              <CustomText style={styles.titleText}>비밀번호 수정하기</CustomText>
            </TouchableOpacity>
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
                // dispatch(setUserInfo({ field: 'password', value: text }));
              }}
              secureTextEntry={securePassword}
              value={password}
            />
            {password && (
                <TouchableOpacity onPress={() => setSecurePassword(!securePassword)} style={styles.passwordClearButton}>
                  <Icon name={securePassword ? 'eye' : 'eye-off'} size={20} color="#B4B4B4" />
                </TouchableOpacity>
              )}
            <View style={styles.passwordCheckContainer}>
              <CustomText style={styles.passwordCheckText}>숫자, 영어 대소문자, 특수문자 포함 필요, 8자 이상</CustomText>
            </View>
            <View style={styles.confirmContainer}>
              {password && !isFormValid() && (
                <CustomText style={password === '' ? styles.confirmTextHidden : styles.confirmText}>비밀번호 형식이 올바르지 않습니다.</CustomText>
              )}

              {/* <CustomText style={password === '' || isFormValid() ? styles.confirmTextHidden : styles.confirmText}>
                비밀번호 형식이 올바르지 않습니다.
              </CustomText> */}
            </View>
          </View>
        </View>

        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'confirmPassword' && styles.labelTextFocused]}>비밀번호 확인</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'confirmPassword' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('confirmPassword')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => setConfirmPassword(text)}
              secureTextEntry={secureConfirmPassword}
              value={confirmPassword}
            />
            {confirmPassword && (
                <TouchableOpacity onPress={() => setSecureConfirmPassword(!secureConfirmPassword)} style={styles.confirmPasswordClearButton}>
                  <Icon name={secureConfirmPassword ? 'eye' : 'eye-off'} size={20} color="#B4B4B4" />
                </TouchableOpacity>
              )}
            <View style={styles.confirmContainer}>
              {confirmPassword && password !== confirmPassword && (
                <CustomText style={confirmPassword === '' ? styles.confirmTextHidden : styles.confirmText}>비밀번호가 일치하지 않습니다.</CustomText>
              )}
            </View>
          </View>
        </View>

        <View>
          <TouchableOpacity onPress={handleUpdate} style={[styles.actionButton, isFormValid() && password === confirmPassword ? styles.actionButtonEnabled : styles.actionButtonDisabled]} disabled={!isFormValid()}>
            <CustomText style={[styles.actionButtonText, isFormValid() ? styles.actionButtonTextEnabled : styles.actionButtonTextDisabled]}>수정 완료</CustomText>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollContent: {
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    position: 'relative',
    height: 50,
  },
  backContainer: {
    paddingHorizontal: 4,
    zIndex: 1,
  },
  section: {
    marginTop: 20,
  },
  title: {
    paddingHorizontal: 5,
    paddingTop: 10,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    // marginBottom: 20,
    marginBottom: 30,
  },
  titleText: {
    fontSize: 24,
  },
  passwordCheckContainer: {
    height: 20,
  },
  passwordCheckText: {
    marginTop: 5,
    color: '#747474',
  },
  confirmContainer: {
    height: 20,
  },
  confirmText: {
    marginTop: 5,
    color: '#D64747',
  },
  confirmTextHidden: {
    marginTop: 5,
    color: 'transparent',
  },
  inputGroup: {
    marginBottom: 40,
    marginHorizontal: 5,
  },
  labelText: {
    fontSize: 18,
    color: '#202020',
  },
  labelTextFocused: {
    color: '#9F8473',
  },
  inputFieldWrapper: {
    // position: 'relative',
    flexDirection: 'column',
    // alignItems: 'center',
  },
  passwordClearButton: {
    position: 'absolute',
    top: '35%',
    right: 1,
    transform: [{ translateY: -8 }],
  },
  confirmPasswordClearButton: {
    position: 'absolute',
    top: '40%',
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
  actionButton: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    marginHorizontal: 5,
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

export default PasswordEditScreen;