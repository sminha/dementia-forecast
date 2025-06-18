import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, TextInput, StyleSheet } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
// import { setUserInfo, updateUser, clearUpdateResult } from '../../redux/slices/userSlice.ts';
import { setBiometricInfo } from '../../redux/slices/biometricSlice.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
// import { loadTokens } from '../../redux/actions/authAction.ts';
import CustomText from '../../components/CustomText.tsx';

const BiometricInputScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const biometricInfo = useSelector((state: RootState) => state.biometric);
  // const userInfo = useSelector((state: RootState) => state.user.userInfo);
  // const updateResult = useSelector((state: RootState) => state.user.updateResult);

  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [focusedField, setFocusedField] = useState<string | null>(null);

  // useEffect(() => {
  //   if (updateResult?.statusCode === 200) {
  //     dispatch(setBiometricInfo({ field: 'password', value: height }));
  //     dispatch(setBiometricInfo({ field: 'password', value: weight }));
  //     dispatch(clearUpdateResult());
  //     navigation.navigate('BiometricSubmitComplete');
  //   } else if (updateResult && updateResult.statusCode !== 200) {
  //     dispatch(clearUpdateResult());
  //   }

  //   // const timer = setTimeout(() => {
  //   //   navigation.navigate('BiometricSubmitComplete');
  //   // }, 3000);

  //   // return () => clearTimeout(timer);
  // }, [updateResult, height, navigation, dispatch]);

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isFormValid = () => {
    return height !== '' && height !== '0' && weight !== '' && weight !== '0';
  };

  const handleUpdate = async () => {
    // const { accessToken } = await loadTokens();
    // if (!accessToken) {
    //   console.log('로그인 정보가 없습니다.');
    //   return;
    // }

    // dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo }, updatedFields: { password } }));
    // dispatch(updateUser({ token: accessToken, userInfo }));

    dispatch(setBiometricInfo({ field: 'height', value: height }));
    dispatch(setBiometricInfo({ field: 'weight', value: weight }));

    navigation.navigate('BiometricSubmitComplete');
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          {/* <TouchableOpacity style={styles.backContainer} onPress={() => navigation.goBack()}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity> */}
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <TouchableOpacity style={styles.titleRow}>
              <CustomText style={styles.titleText}>신장 및 체중 입력하기</CustomText>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'height' && styles.labelTextFocused]}>신장</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'height' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('height')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => {
                setHeight(text);
                // dispatch(setUserInfo({ field: 'password', value: text }));
              }}
              value={height}
              keyboardType="numeric"
              placeholder="cm 단위로 입력하세요."
            />
          </View>
        </View>

        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'weight' && styles.labelTextFocused]}>체중</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'weight' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('weight')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => setWeight(text)}
              value={weight}
              keyboardType="numeric"
              placeholder="kg 단위로 입력하세요."
            />
          </View>
        </View>
      </ScrollView>
      <View style={styles.actionButtonWrapper}>
          <TouchableOpacity onPress={handleUpdate} style={[styles.actionButton, isFormValid() ? styles.actionButtonEnabled : styles.actionButtonDisabled]} disabled={!isFormValid()}>
            <CustomText style={[styles.actionButtonText, isFormValid() ? styles.actionButtonTextEnabled : styles.actionButtonTextDisabled]}>수정 완료</CustomText>
          </TouchableOpacity>
        </View>
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
    // height: 50,
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
  actionButtonWrapper: {
    paddingVertical: 10,
    paddingHorizontal: 16,
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

export default BiometricInputScreen;