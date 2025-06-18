import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, StyleSheet } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../../types/navigationTypes.ts';
import { setUserInfo, updateUser, clearUpdateResult } from '../../../redux/slices/userSlice.ts';
import { AppDispatch, RootState } from '../../../redux/store.ts';
import { loadTokens } from '../../../redux/actions/authAction.ts';
import CustomText from '../../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const GenderEditScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const updateResult = useSelector((state: RootState) => state.user.updateResult);

  const [gender, setGender] = useState(userInfo.gender);
  // const [focusedField, setFocusedField] = useState<string | null>(null);

  useEffect(() => {
    // if (userInfo.gender) {
    //   setGender(userInfo.gender);
    // }

    // if (updateResult?.statusCode === 200) {
    if (updateResult?.message === '회원 정보가 수정되었습니다.') {
      dispatch(setUserInfo({ field: 'gender', value: gender }));
      dispatch(clearUpdateResult());
      navigation.goBack();
    } else if (updateResult && updateResult.message !== '회원 정보가 수정되었습니다.') {
      dispatch(clearUpdateResult());
    }
  }, [updateResult, gender, navigation, dispatch]);

  // const handleFocus = (field: string) => {
  //   setFocusedField(field);
  // };

  // const handleBlur = () => {
  //   setFocusedField(null);
  // };

  const isFormValid = () => {
    return gender && gender !== userInfo.gender;
  };

  const handleUpdate = async () => {
    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    // dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo }, updatedFields: { gender } }));
    dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo, gender } }));
    // dispatch(updateUser({ token: accessToken, userInfo }));
  };

  // const handleChange = (field: string, value: string | boolean) => {
  //   dispatch(setUserInfo({ field, value }));
  // };

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
              <CustomText style={styles.titleText}>성별 수정하기</CustomText>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.rowWithoutMatch}>
          <CustomText style={[styles.label, /*focusedField === 'gender' &&*/ styles.focusedLabel]}>성별</CustomText>
          <View style={styles.genderButtonContainer}>
            <TouchableOpacity
              style={[styles.genderButtonLeft, gender === '남' && styles.genderButtonSelected]}
              onPress={() => setGender('남') /*dispatch(setUserInfo({ field: 'gender', value: '남' }))*/}
            >
              <CustomText style={[styles.genderButtonText, gender === '남' && styles.genderButtonTextSelected]}>남</CustomText>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.genderButtonRight, gender === '여' && styles.genderButtonSelected]}
              onPress={() => setGender('여') /*dispatch(setUserInfo({ field: 'gender', value: '여' }))*/}
            >
              <CustomText style={[styles.genderButtonText, gender === '여' && styles.genderButtonTextSelected]}>여</CustomText>
            </TouchableOpacity>
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
  rowWithoutMatch: {
    marginBottom: 60,
    marginHorizontal: 5,
  },
  label: {
    fontSize: 18,
    color: '#202020',
  },
  focusedLabel: {
    color: '#9F8473',
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

export default GenderEditScreen;