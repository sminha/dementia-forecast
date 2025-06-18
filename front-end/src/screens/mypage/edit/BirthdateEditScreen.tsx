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

const formatBirthdate = (input: string): string => {
  const yearPrefix = Number(input.slice(0, 2)) > 30 ? '19' : '20';

  const fullDateStr = yearPrefix + input; // 예: '19591111' 또는 '20010101'

  return fullDateStr;
  // return Number(fullDateStr);
};

function convertShortBirthToNumber(birth: string): number {
  const yearPrefix = parseInt(birth.slice(0, 2), 10) <= 24 ? '20' : '19';
  const fullDate = `${yearPrefix}${birth}`;
  return parseInt(fullDate, 10);
}

const toISOStringFromYYMMDD = (input: string): string => {
  const yearNum = Number(input.slice(0, 2));
  const yearPrefix = yearNum > 25 ? '19' : '20'; // 31 이상은 1900년대, 30 이하는 2000년대 기준
  const year = yearPrefix + input.slice(0, 2);
  const month = input.slice(2, 4);
  const day = input.slice(4, 6);

  const date = new Date(`${year}-${month}-${day}T00:00:00.000Z`);
  return date.toISOString();
};

const BirthdateEditScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const updateResult = useSelector((state: RootState) => state.user.updateResult);

  const [birthdate, setBirthdate] = useState('');
  const [focusedField, setFocusedField] = useState<string | null>(null);

  const [isoBirthdate, setIsoBirthdate] = useState('');

  useEffect(() => {
    // if (updateResult?.statusCode === 200) {
    if (updateResult?.message === '회원 정보가 수정되었습니다.') {
      dispatch(setUserInfo({ field: 'birthdate', value: isoBirthdate }));
      dispatch(clearUpdateResult());
      navigation.goBack();
    } else if (updateResult && updateResult.message !== '회원 정보가 수정되었습니다.') {
      dispatch(clearUpdateResult());
    }
  }, [updateResult, isoBirthdate, navigation, dispatch]);

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isFormValid = () => {
    return /^\d{6}$/.test(birthdate) && birthdate !== userInfo.birthdate;
  };

  const handleUpdate = async () => {
    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    // isoBirthdate = toISOStringFromYYMMDD(birthdate);
    // setIsoBirthdate(toISOStringFromYYMMDD(birthdate));

    const converted = toISOStringFromYYMMDD(birthdate);
    setIsoBirthdate(converted);

    // dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo }, updatedFields: { birthdate } }));
    dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo, birthdate: isoBirthdate } }));
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
              <CustomText style={styles.titleText}>생년월일 수정하기</CustomText>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.inputGroup}>
          <CustomText style={[styles.labelText, focusedField === 'birthdate' && styles.labelTextFocused]}>생년월일</CustomText>
          <View style={styles.inputFieldWrapper}>
            <TextInput
              style={[styles.inputField, focusedField === 'birthdate' && styles.inputFieldFocused]}
              onFocus={() => handleFocus('birthdate')}
              onBlur={() => handleBlur()}
              onChangeText={(text) => {
                setBirthdate(text);
                // dispatch(setUserInfo({ field: 'birthdate', value: text }));
              }}
              value={birthdate}
              keyboardType="number-pad"
              placeholder="6자리"
            />
            <View style={styles.confirmContainer}>
              {birthdate && !isFormValid() && (
                <CustomText style={birthdate === '' ? styles.confirmTextHidden : styles.confirmText}>생년월일은 6자리 숫자여야 합니다.</CustomText>
              )}
            </View>
            {birthdate !== '' &&
            <TouchableOpacity onPress={() => setBirthdate('')} style={styles.clearButton}>
              <Icon name="close-circle" size={20} color="#B4B4B4" />
            </TouchableOpacity>
            }
          </View>
        </View>

        <View>
          <TouchableOpacity onPress={handleUpdate} style={[styles.actionButton, isFormValid() ? styles.actionButtonEnabled : styles.actionButtonDisabled]} disabled={!isFormValid()}>
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
  clearButton: {
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

export default BirthdateEditScreen;