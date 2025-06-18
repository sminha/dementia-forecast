import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, Switch, StyleSheet } from 'react-native';
import Modal from 'react-native-modal';
import { useDispatch, useSelector } from 'react-redux';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { clearLoginResult } from '../../redux/slices/loginSlice.ts';
import { clearFetchResult } from '../../redux/slices/lifestyleSlice.ts';
import { logout, logoutUser, clearLogoutResult } from '../../redux/slices/userSlice.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import { loadTokens } from '../../redux/actions/authAction.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const MypageScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const name = useSelector((state: RootState) => state.user.userInfo.name);
  const logoutResult = useSelector((state: RootState) => state.user.logoutResult);

  const userInfo = useSelector((state: RootState) => state.user.userInfo);

  const [isLargeText, setIsLargeText] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isBiometricSaved, setIsBiometricSaved] = useState(false);
  const [backConfirmModal, setBackConfirmModal] = useState(false);
  const [biometricSavedModal, setBiometricSavedModal] = useState(false);

  useEffect(() => {
    const handleLogoutSuccess = async () => {
      if (logoutResult?.message === '로그아웃 성공') {
        // const accessToken1 = await AsyncStorage.getItem('accessToken');
        // const refreshToken1 = await AsyncStorage.getItem('refreshToken');
        // console.log('accessToken:', accessToken1, 'refreshToken:', refreshToken1);
        // console.log(userInfo);

        await AsyncStorage.removeItem('accessToken');
        await AsyncStorage.removeItem('refreshToken');
        await AsyncStorage.removeItem('loggedInUser');
        // await AsyncStorage.removeItem('biometricInfo');
        // const accessToken2 = await AsyncStorage.getItem('accessToken');
        // const refreshToken2 = await AsyncStorage.getItem('refreshToken');
        // console.log('accessToken:', accessToken2, 'refreshToken:', refreshToken2);

        dispatch(logout());
        dispatch(clearLoginResult());
        dispatch(clearLogoutResult());
        dispatch(clearFetchResult());
        navigation.reset({
          index: 0,
          routes: [{name: 'Login'}],
        });
      } else {
        dispatch(clearLogoutResult());
      }
    };

    handleLogoutSuccess();
  }, [logoutResult, dispatch, navigation, userInfo]);

  useEffect(() => {
    const fetchBiometricInfo = async () => {
      const biometricInfo = await AsyncStorage.getItem('biometricInfo');
      if (biometricInfo) {
        const parsed = JSON.parse(biometricInfo);
        if (parsed?.email === userInfo.email) {
          setIsBiometricSaved(true);
        } else {
          setIsBiometricSaved(false);
        }
      } else {
        setIsBiometricSaved(false);
      }
    };

    fetchBiometricInfo();
  }, [userInfo.email]);

  const handleLogout = async () => {
    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    // dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo }, updatedFields: { birthdate } }));
    dispatch(logoutUser(accessToken));
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => navigation.replace('Home')}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
          <View style={styles.titleWrapper}>
            <CustomText style={styles.loginText}>마이페이지</CustomText>
          </View>
        </View>

        <View style={styles.section}>
          <TouchableOpacity
            style={styles.row}
            onPress={() => navigation.navigate('AccountView')}
          >
            <CustomText style={styles.sectionTitle}>{name}님, 안녕하세요!</CustomText>
            <Icon name="chevron-forward" size={16} color="gray" />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <CustomText weight="bold" style={styles.sectionTitle}>나의 기록</CustomText>
          <View style={styles.list}>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleView')}>
              <CustomText style={styles.listText}>라이프스타일</CustomText>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => {isBiometricSaved ? navigation.navigate('BiometricOverview', { from: 'Mypage' }) : setBiometricSavedModal(true)}}>
              <CustomText style={styles.listText}>생체정보</CustomText>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('ReportView', { from: 'Mypage' })}>
              <CustomText style={styles.listText}>진단 결과</CustomText>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
          </View>
        </View>

        {/* <View style={styles.section}>
          <CustomText weight="bold" style={styles.sectionTitle}>모드 설정</CustomText>
          <View style={styles.list}>
            <TouchableOpacity style={styles.listRow}>
              <CustomText style={styles.listText}>큰글 모드</CustomText>
              <Switch
                value={isLargeText}
                onValueChange={setIsLargeText}
                trackColor={{ false: '#D6CCC2', true: '#917A6B' }}
                thumbColor={'#FFFFFF'}
              />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow}>
              <CustomText style={styles.listText}>다크 모드</CustomText>
              <Switch
                value={isDarkMode}
                onValueChange={setIsDarkMode}
                trackColor={{ false: '#D6CCC2', true: '#917A6B' }}
                thumbColor={'#FFFFFF'}
              />
            </TouchableOpacity>
          </View>
        </View> */}

        <View style={styles.section}>
          <CustomText weight="bold" style={styles.sectionTitle}>계정 관리</CustomText>
          <View style={styles.list}>
            <TouchableOpacity style={styles.listRow} onPress={() => setBackConfirmModal(true)}>
              <CustomText style={styles.listText}>로그아웃</CustomText>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('AccountDelete')}>
              <CustomText style={styles.listText}>회원탈퇴</CustomText>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
          </View>
        </View>

        <Modal
        isVisible={backConfirmModal}
        onBackdropPress={() => setBackConfirmModal(false)}
        onBackButtonPress={() => setBackConfirmModal(false)}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
        >
          <View style={styles.modalContent}>
            <CustomText style={styles.modalTitle}>정말 로그아웃하시겠어요?</CustomText>

            <View style={styles.modalButtonWrapper}>
              <TouchableOpacity style={[styles.modalButton, styles.modalButtonPrimary]} onPress={() => setBackConfirmModal(false)}>
                <CustomText style={styles.modalButtonText}>아니요, 안 할래요</CustomText>
              </TouchableOpacity>

              <TouchableOpacity style={[styles.modalButton, styles.modalButtonSecondary]} onPress={() => handleLogout()}>
                <CustomText style={styles.modalButtonText}>네, 로그아웃 할래요</CustomText>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>

        <Modal
        isVisible={biometricSavedModal}
        onBackdropPress={() => setBiometricSavedModal(false)}
        onBackButtonPress={() => setBiometricSavedModal(false)}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
        >
          <View style={styles.modalContent}>
            <CustomText style={styles.modalTitle}>입력된 생체정보가 없어요.</CustomText>

            <View style={styles.modalButtonWrapper}>
              <TouchableOpacity style={[styles.modalButton, styles.modalButtonPrimary]} onPress={() => setBiometricSavedModal(false)}>
                <CustomText style={styles.modalButtonText}>나가기</CustomText>
              </TouchableOpacity>

              <TouchableOpacity style={[styles.modalButton, styles.modalButtonSecondary]} onPress={() => navigation.navigate('BiometricStart')}>
                <CustomText style={styles.modalButtonText}>입력하러 가기</CustomText>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
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
  titleWrapper: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loginText: {
    fontSize: 24,
  },
  section: {
    marginTop: 20,
  },
  sectionSecondary: {
    marginTop: 30,
  },
  sectionTitle: {
    fontSize: 24,
    marginBottom: 8,
  },
  list: {
    paddingHorizontal: 5,
    paddingTop: 10,
  },
  listText: {
    fontSize: 18,
    color: '#434240',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingRight: 5,
  },
  listRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  modalOverlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalContent: {
    alignItems: 'center',
    width: '90%',
    padding: 10,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
  },
  modalTitle: {
    marginVertical: 40,
    fontSize: 22,
  },
  modalButtonWrapper: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
  },
  modalButton: {
    flex: 1,
    alignItems: 'center',
    padding: 10,
    borderRadius: 10,
  },
  modalButtonPrimary: {
    backgroundColor: '#F2EAE3',
  },
  modalButtonSecondary: {
    backgroundColor: '#D6CCC2',
  },
  modalButtonText: {
    fontSize: 18,
    color: '#575553',
  },
  modalContentText: {
    fontSize: 22,
    margin: 50,
  },
});

export default MypageScreen;