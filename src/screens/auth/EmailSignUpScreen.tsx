import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Pressable, Keyboard, View, SafeAreaView, TouchableOpacity, ScrollView, TextInput, StyleSheet } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import Modal from 'react-native-modal';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import Postcode from '@actbase/react-daum-postcode';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { setFormData, clearRegistrationResult, registerUser } from '../../redux/slices/signupSlice.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const EmailSignUpScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'EmailSignUp'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const formData = useSelector((state: RootState) => state.signup.formData);
  const isLoading = useSelector((state: RootState) => state.signup.isLoading);
  const registrationResult = useSelector((state: RootState) => state.signup.registrationResult);

  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [securePassword, setSecurePassword] = useState(true);
  const [secureConfirmPassword, setSecureConfirmPassword] = useState(true);
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isPostcodeVisible, setPostcodeVisible] = useState(false);
  // const [isErrorModalVisible, setErrorModalVisible] = useState(false);

  const isFirstFocus = useRef(true);
  const emailInputRef = useRef<TextInput>(null);

  useFocusEffect(
    useCallback(() => {
      if (isFirstFocus.current) {
        dispatch(setFormData({ field: 'email', value: '' }));
        dispatch(setFormData({ field: 'password', value: '' }));
        dispatch(setFormData({ field: 'name', value: '' }));
        dispatch(setFormData({ field: 'gender', value: '' }));
        dispatch(setFormData({ field: 'birthdate', value: '' }));
        dispatch(setFormData({ field: 'phone', value: '' }));
        dispatch(setFormData({ field: 'address', value: '' }));
        dispatch(setFormData({ field: 'detailAddress', value: '' }));
        dispatch(setFormData({ field: 'agreeAll', value: false }));
        dispatch(setFormData({ field: 'privacyShare', value: false }));
        dispatch(setFormData({ field: 'privacyUse', value: false }));
        isFirstFocus.current = false;
      } else {
        if (formData.privacyShare && formData.privacyUse) {
          dispatch(setFormData({ field: 'agreeAll', value: true }));
        } else {
          dispatch(setFormData({ field: 'agreeAll', value: false }));
        }
      }

      return () => {
        dispatch(clearRegistrationResult());
      };
    }, [dispatch, formData.privacyShare, formData.privacyUse])
  );


  useEffect(() => {
    if (registrationResult?.statusCode === 200) {
      navigation.navigate('EmailSignUpComplete');
    } else if (registrationResult?.message === '이미 존재하는 이메일입니다.')  {
      setTimeout(() => {
        emailInputRef.current?.focus();
      }, 100);
    } else {
      // setErrorModalVisible(true);
    }
  }, [registrationResult, navigation]);

  // useEffect(() => {
  //   const timer = setTimeout(() => {
  //     setErrorModalVisible(false);
  //   }, 3000);

  //   return () => clearTimeout(timer);
  // }, [isErrorModalVisible]);

  const handleChange = (field: string, value: string | boolean) => {
    if (field === 'agreeAll') {
      dispatch(setFormData({ field: 'agreeAll', value }));
      dispatch(setFormData({ field: 'privacyShare', value }));
      dispatch(setFormData({ field: 'privacyUse', value }));
    } else {
      dispatch(setFormData({ field, value }));
    }

    if ((field === 'privacyShare' || field === 'privacyUse') && value === false) {
      dispatch(setFormData({ field: 'agreeAll', value: false }));
    }

    if ((field === 'privacyShare' || field === 'privacyUse')) {
      const otherField = field === 'privacyShare' ? 'privacyUse' : 'privacyShare';
      const otherValue = (formData as any)[otherField];
      if (value === true && otherValue === true) {
        dispatch(setFormData({ field: 'agreeAll', value: true }));
      }
    }
  };

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isEmailValid = (email: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };

  const isPasswordValid = (password: string) => {
    const lengthValid = password.length >= 8;
    const hasUpper = /[A-Z]/.test(password);
    const hasLower = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[^A-Za-z0-9]/.test(password);

    const conditionsMet = [hasUpper, hasLower, hasNumber, hasSpecial].filter(Boolean).length >= 4;

    return lengthValid && conditionsMet;
  };

  const isPasswordMatch = (password: string, passwordCheck: string) => {
    return password === passwordCheck;
  };

  const isBirthdateValid = (birthdate: string) => {
    return /^\d{6}$/.test(birthdate);
  };

  const isPhoneValid = (phone: string) => {
    return /^01[0-9]{9}$/.test(phone);
  };

  const isFormValid = () => {
    return (
      formData.email &&
      formData.password &&
      formData.name &&
      formData.gender &&
      formData.birthdate &&
      formData.phone &&
      formData.address &&
      formData.detailAddress &&
      formData.privacyShare &&
      formData.privacyUse &&
      isEmailValid(formData.email) &&
      isPasswordValid(formData.password) &&
      isPasswordMatch(formData.password, confirmPassword) &&
      isBirthdateValid(formData.birthdate) &&
      isPhoneValid(formData.phone)
    );
  };

  const handleSignUp = () => {
    if (isFormValid()) {
      dispatch(registerUser(formData));
    }
  };

  return (
    <Pressable onPress={() => {
      Keyboard.dismiss();
      setSecurePassword(true);
      setSecureConfirmPassword(true);
      }}
      style={styles.pressableContainer}
    >
      <View style={styles.container}>
        <SafeAreaView style={styles.safeArea}>
          <TouchableOpacity onPress={() => navigation.navigate('Login')}>
            <Icon name="chevron-back" size={16} />
          </TouchableOpacity>
        </SafeAreaView>

        <View style={styles.header}>
          <CustomText style={styles.title}>이메일로 회원가입하기</CustomText>
        </View>

        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
          <View style={styles.row}>
            <CustomText style={[styles.label, focusedField === 'email' && styles.focusedLabel]}>이메일</CustomText>
            <View style={styles.inputFieldWrapper}>
              <TextInput
                style={[styles.input, focusedField === 'email' && styles.focusedInput]}
                onFocus={() => handleFocus('email')}
                onBlur={() => handleBlur()}
                onChangeText={(text) => handleChange('email', text)}
                ref={emailInputRef}
                value={formData.email}
              />
              {formData.email !== '' &&
                <TouchableOpacity onPress={() => handleChange('email', '')} style={styles.clearButton}>
                  <Icon name="close-circle" size={20} color="#B4B4B4" />
                </TouchableOpacity>
              }
            </View>
            <View style={styles.confirmContainer}>
              {formData.email && !isEmailValid(formData.email) && (
                <CustomText style={formData.email === '' ? styles.confirmTextHidden : styles.confirmText}>이메일 형식이 올바르지 않습니다.</CustomText>
              )}
              {registrationResult?.message && (registrationResult.message === '이미 존재하는 이메일입니다.') && (
                <CustomText style={styles.confirmText}>이미 존재하는 이메일입니다.</CustomText>
              )}
            </View>
          </View>

          <View style={styles.row}>
            <CustomText style={[styles.label, focusedField === 'password' && styles.focusedLabel]}>비밀번호</CustomText>
            <View style={styles.inputFieldWrapper}>
              <TextInput
                style={[styles.input, focusedField === 'password' && styles.focusedInput]}
                onFocus={() => handleFocus('password')}
                onBlur={() => {
                  handleBlur();
                  setSecurePassword(true);
                }}
                onChangeText={(text) => handleChange('password', text)}
                secureTextEntry={securePassword}
                value={formData.password}
              />
              {formData.password !== '' && (
                <TouchableOpacity onPress={() => setSecurePassword(!securePassword)} style={styles.clearButton}>
                  <Icon name={securePassword ? 'eye' : 'eye-off'} size={20} color="#B4B4B4" />
                </TouchableOpacity>
              )}
            </View>
            <View style={styles.passwordCheckContainer}>
              <CustomText style={styles.passwordCheckText}>숫자, 영어 대소문자, 특수문자 포함 필요, 8자 이상</CustomText>
            </View>
            <View style={styles.confirmContainer}>
              {formData.password && !isPasswordValid(formData.password) && (
                <CustomText style={formData.password === '' ? styles.confirmTextHidden : styles.confirmText}>비밀번호 형식이 올바르지 않습니다.</CustomText>
              )}
            </View>
          </View>

          <View style={styles.row}>
            <CustomText style={[styles.label, focusedField === 'confirmPassword' && styles.focusedLabel]}>비밀번호 확인</CustomText>
            <View style={styles.inputFieldWrapper}>
              <TextInput
                style={[styles.input, focusedField === 'confirmPassword' && styles.focusedInput]}
                onFocus={() => handleFocus('confirmPassword')}
                onBlur={() => {
                  handleBlur();
                  setSecureConfirmPassword(true);
                }}
                onChangeText={(text) => setConfirmPassword(text)}
                secureTextEntry={secureConfirmPassword}
                value={confirmPassword}
              />
              {confirmPassword !== '' && (
                <TouchableOpacity onPress={() => setSecureConfirmPassword(!secureConfirmPassword)} style={styles.clearButton}>
                  <Icon name={secureConfirmPassword ? 'eye' : 'eye-off'} size={20} color="#B4B4B4" />
                </TouchableOpacity>
              )}
            </View>

            <View style={styles.confirmContainer}>
              {confirmPassword && !isPasswordMatch(formData.password, confirmPassword) && (
                <CustomText style={confirmPassword === '' ? styles.confirmTextHidden : styles.confirmText}>비밀번호가 일치하지 않습니다.</CustomText>
              )}
            </View>
          </View>

          <View style={styles.rowWithoutMatch}>
            <CustomText style={[styles.label, focusedField === 'name' && styles.focusedLabel]}>이름</CustomText>
            <TextInput
              style={[styles.input, focusedField === 'name' && styles.focusedInput]}
              onFocus={() => handleFocus('name')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => handleChange('name', text)}
          />
          </View>

          <View style={styles.rowWithoutMatch}>
            <CustomText style={[styles.label, focusedField === 'gender' && styles.focusedLabel]}>성별</CustomText>
            <View style={styles.genderButtonContainer}>
              <TouchableOpacity
                style={[styles.genderButtonLeft, formData.gender === '남' && styles.genderButtonSelected]}
                onPress={() => handleChange('gender', '남')}
              >
                <CustomText style={[styles.genderButtonText, formData.gender === '남' && styles.genderButtonTextSelected]}>남</CustomText>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.genderButtonRight, formData.gender === '여' && styles.genderButtonSelected]}
                onPress={() => handleChange('gender', '여')}
              >
                <CustomText style={[styles.genderButtonText, formData.gender === '여' && styles.genderButtonTextSelected]}>여</CustomText>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.row}>
            <CustomText style={[styles.label, focusedField === 'birthdate' && styles.focusedLabel]}>생년월일</CustomText>
            <TextInput
              style={[styles.input, focusedField === 'birthdate' && styles.focusedInput]}
              onFocus={() => handleFocus('birthdate')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => handleChange('birthdate', text)}
              keyboardType="number-pad"
              placeholder="6자리"
            />
            <View style={styles.confirmContainer}>
                {formData.birthdate && !isBirthdateValid(formData.birthdate) && (
                  <CustomText style={formData.birthdate === '' ? styles.confirmTextHidden : styles.confirmText}>생년월일은 6자리 숫자여야 합니다.</CustomText>
                )}
              </View>
          </View>

          <View style={styles.row}>
            <CustomText style={[styles.label, focusedField === 'phone' && styles.focusedLabel]}>전화번호</CustomText>
            <TextInput
              style={[styles.input, focusedField === 'phone' && styles.focusedInput]}
              onFocus={() => handleFocus('phone')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => handleChange('phone', text)}
              keyboardType="numeric"
            />
            <View style={styles.confirmContainer}>
                {formData.phone && !isPhoneValid(formData.phone) && (
                  <CustomText style={formData.phone === '' ? styles.confirmTextHidden : styles.confirmText}>전화번호는 11자리 숫자여야 합니다.</CustomText>
                )}
              </View>
          </View>

          <View style={styles.row}>
            <CustomText style={[styles.label, focusedField === 'address' && styles.focusedLabel]}>주소</CustomText>
            <TouchableOpacity
              onPress={() => setPostcodeVisible(true)}
              style={styles.addressInput}
            >
              <CustomText style={styles.addressText}>{formData.address}</CustomText>
            </TouchableOpacity>
            <TextInput
              style={[styles.detailAddressInput, focusedField === 'detailAddress' && styles.focusedInput]}
              onFocus={() => handleFocus('detailAddress')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => handleChange('detailAddress', text)}
            />
          </View>

          <View style={styles.checkboxGroupConainer}>
          <View style={styles.checkboxContainer}>
            <Pressable onPress={() => handleChange('agreeAll', !formData.agreeAll)}>
              <Icon
                name={formData.agreeAll ? 'checkmark-circle' : 'checkmark-circle-outline'}
                size={24}
                color={formData.agreeAll ? '#9F8473' : '#B4B4B4'}
              />
            </Pressable>
            <CustomText style={styles.checkboxText}>전체동의</CustomText>
          </View>

          <View style={styles.separator} />

          <View style={styles.checkboxContainer}>
            <Pressable onPress={() => handleChange('privacyUse', !formData.privacyUse)}>
              <Icon
                name={formData.privacyUse ? 'checkmark-circle' : 'checkmark-circle-outline'}
                size={24}
                color={formData.privacyUse ? '#9F8473' : '#B4B4B4'}
              />
            </Pressable>
            <CustomText style={styles.checkboxText}>개인정보 수집 및 이용 동의</CustomText>
            <TouchableOpacity onPress={() => navigation.navigate('PrivacyUsePolicy')}>
              <Icon name="chevron-forward" size={16} color="gray" style={styles.chevronIcon} />
            </TouchableOpacity>
          </View>

          <View style={styles.checkboxContainer}>
            <Pressable onPress={() => handleChange('privacyShare', !formData.privacyShare)}>
              <Icon
                name={formData.privacyShare ? 'checkmark-circle' : 'checkmark-circle-outline'}
                size={24}
                color={formData.privacyShare ? '#9F8473' : '#B4B4B4'}
              />
            </Pressable>
            <CustomText style={styles.checkboxText}>개인정보 제공 동의</CustomText>
            <TouchableOpacity onPress={() => navigation.navigate('PrivacySharePolicy')}>
              <Icon name="chevron-forward" size={16} color="gray" style={styles.chevronIcon} />
            </TouchableOpacity>
          </View>
        </View>
        </ScrollView>

        <View>
          <TouchableOpacity
            onPress={handleSignUp}
            style={[styles.button, isFormValid() && !isLoading ? styles.buttonAbled : styles.buttonDisabled]}
            disabled={!isFormValid() || isLoading}
          >
            <CustomText style={[styles.buttonText, isFormValid() && !isLoading ? styles.buttonTextAbled : styles.buttonTextDisabled]}>
              {isLoading ? '회원가입 중...' : '회원가입'}
            </CustomText>
          </TouchableOpacity>

          {/* {registrationResult?.message && (
            <CustomText>{registrationResult.message}</CustomText>
          )} */}
        </View>

        <Modal
          isVisible={isPostcodeVisible}
          onBackdropPress={() => setPostcodeVisible(false)}
          onBackButtonPress={() => setPostcodeVisible(false)}
          backdropColor="rgb(69, 69, 69)"
          backdropOpacity={0.3}
          animationIn="fadeIn"
          animationOut="fadeOut"
        >
          <View style={styles.postcodeContainer}>
            <Postcode
              style={styles.postcode}
              jsOptions={{ animation: true }}
              onSelected={(data) => {
                handleChange('address', data.address);
                setPostcodeVisible(false);
              }}
              onError={() => setPostcodeVisible(false)}
            />
          </View>
        </Modal>

        {/* <Modal
          isVisible={isErrorModalVisible}
          backdropColor="rgb(69, 69, 69)"
          backdropOpacity={0.3}
          animationIn="fadeIn"
          animationOut="fadeOut"
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <CustomText style={styles.modalTitle}>서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.</CustomText>
            </View>
          </View>
        </Modal> */}
      </View>
    </Pressable>
  );
};

const styles = StyleSheet.create({
  pressableContainer: {
    flex: 1,
  },
  container: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    backgroundColor: '#FFFFFF',
  },
  safeArea: {
    width: 70,
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  header: {
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  backButton: {
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
  },
  scrollContent: {
    paddingHorizontal: 20,
  },
  row: {
    marginBottom: 40,
  },
  rowWithoutMatch: {
    marginBottom: 60,
  },
  label: {
    fontSize: 18,
    color: '#202020',
  },
  focusedLabel: {
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
  input: {
    width: '100%',
    paddingRight: 35,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  focusedInput: {
    borderBottomColor: '#9F8473',
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
  genderButtonContainer: {
    flexDirection: 'row',
  },
  genderButtonLeft: {
    flex: 50,
    alignItems: 'center',
    padding: 10,
    marginTop: 20,
    marginRight: 7,
    borderWidth: 1,
    borderColor: '#B4B4B4',
  },
  genderButtonRight: {
    flex: 50,
    alignItems: 'center',
    padding: 10,
    marginTop: 20,
    marginLeft: 7,
    borderWidth: 1,
    borderColor: '#B4B4B4',
  },
  genderButtonSelected: {
    borderColor: '#D6CCC2',
    backgroundColor: '#D6CCC2',
  },
  genderButtonText: {
    fontSize: 18,
    color: '#6C6C6B',
  },
  genderButtonTextSelected: {
    color: '#575553',
  },
  addressInput: {
    height: 45,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    justifyContent: 'center',
  },
  addressText: {
    marginLeft: 4,
    fontSize: 18,
  },
  detailAddressInput: {
    marginTop: 20,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  checkboxGroupConainer: {
    marginTop: 10,
  },
  checkboxContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  checkboxText: {
    flex: 1,
    marginLeft: 10,
    fontSize: 20,
    lineHeight: 24,
  },
  separator: {
    height: 1,
    marginTop: 5,
    marginBottom: 20,
    backgroundColor: '#D5D5D5',
  },
  chevronIcon: {
    marginLeft: 'auto',
  },
  button: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    borderRadius: 5,
  },
  buttonAbled: {
    backgroundColor: '#D6CCC2',
  },
  buttonDisabled: {
    backgroundColor: '#F2EFED',
  },
  buttonText: {
    fontSize: 20,
  },
  buttonTextAbled: {
    color: '#575553',
  },
  buttonTextDisabled: {
    color: '#B4B4B4',
  },
  postcodeContainer: {
    flex: 0.7,
  },
  postcode: {
    flex: 1,
  },
  // modalOverlay: {
  //   flex: 1,
  //   alignItems: 'center',
  //   justifyContent: 'center',
  //   backgroundColor: 'rgba(69, 69, 69, 0.3)',
  // },
  // modalContent: {
  //   alignItems: 'center',
  //   width: '90%',
  //   padding: 10,
  //   borderRadius: 10,
  //   backgroundColor: '#FFFFFF',
  // },
  // modalTitle: {
  //   marginVertical: 40,
  //   fontSize: 24,
  // },
});

export default EmailSignUpScreen;