import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, StyleSheet } from 'react-native';
import Modal from 'react-native-modal';
import { useDispatch, useSelector } from 'react-redux';
import jwt_decode from 'jwt-decode';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { clearLoginResult } from '../../redux/slices/loginSlice.ts';
import { deleteAction, deleteUser, clearDeleteResult, clearLogoutResult, clearFetchResult, deleteLifestyleData, deletePredictionReport } from '../../redux/slices/userSlice.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import { loadTokens } from '../../redux/actions/authAction.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const AccountDeleteScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const deleteResult = useSelector((state: RootState) => state.user.deleteResult);

  const userInfo = useSelector((state: RootState) => state.user.userInfo);

  const [backConfirmModal, setBackConfirmModal] = useState(false);

  useEffect(() => {
    const handleDeleteSuccess = async () => {
      if (deleteResult?.message === '회원 탈퇴 성공') {
        // const accessToken1 = await AsyncStorage.getItem('accessToken');
        // const refreshToken1 = await AsyncStorage.getItem('refreshToken');
        // console.log('accessToken:', accessToken1, 'refreshToken:', refreshToken1);
        // console.log(userInfo);

        await AsyncStorage.removeItem('accessToken');
        await AsyncStorage.removeItem('refreshToken');
        await AsyncStorage.removeItem('loggedInUser');
        await AsyncStorage.removeItem('biometricInfo');
        await AsyncStorage.removeItem('createDate');
        // const accessToken2 = await AsyncStorage.getItem('accessToken');
        // const refreshToken2 = await AsyncStorage.getItem('refreshToken');
        // console.log('accessToken:', accessToken2, 'refreshToken:', refreshToken2);
        // console.log(userInfo);

        dispatch(deleteAction());
        dispatch(clearLoginResult());
        dispatch(clearLogoutResult());
        dispatch(clearFetchResult());
        navigation.reset({
          index: 0,
          routes: [{name: 'Login'}],
        });
        dispatch(clearDeleteResult());
      } else {
        dispatch(clearDeleteResult());
      }
    };

    handleDeleteSuccess();
  }, [deleteResult, dispatch, navigation, userInfo]);

  const handleDelete = async () => {
    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    // const decoded = jwt_decode(accessToken);

    dispatch(deleteUser(accessToken));

  // try {
  //   // 1. 라이프스타일 삭제
  //   await dispatch(deleteLifestyleData(decoded.userid)).unwrap();

  //   // 2. 예측 리포트 삭제
  //   await dispatch(deletePredictionReport(decoded.userid)).unwrap();

  //   // 3. 최종 회원 탈퇴
  //   dispatch(deleteUser(accessToken));
  // } catch (err) {
  //   console.log('회원 탈퇴 중 일부 API 호출 실패:', err);
  //   // 필요시 에러 모달 띄우기
  // }
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
          <TouchableOpacity
            style={styles.row}
            onPress={() => navigation.navigate('AccountView')}
          >
            <CustomText style={styles.sectionTitle}>탈퇴하기 전 꼭 확인하세요!</CustomText>
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <View style={styles.list}>
            <CustomText style={styles.listText}>•  탈퇴 후에도 재가입은 언제든지 가능해요.</CustomText>
            <CustomText style={styles.listText}>•  하지만 탈퇴와 동시에 계정 정보, 진단 결과, 라이프스타일 등의 정보는 영구 삭제되고 삭제된 정보는 다시 복원할 수 없어요.</CustomText>
            <CustomText style={styles.listText}>•  따라서 기존 진단 결과에서 고위험군으로 판정되었더라도 관련 공공기관으로부터 지원을 받지 못 할 수 있어요.</CustomText>
          </View>
        </View>
      </ScrollView>

      <View style={styles.buttonWrapper}>
          <TouchableOpacity
            style={[styles.authButton, styles.authButtonPrimary]}
            onPress={() => navigation.goBack()}
          >
            <CustomText style={styles.authButtonText}>계속 이용하기</CustomText>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.authButton, styles.authButtonSecondary]}
            onPress={() => setBackConfirmModal(true)}
          >
            <CustomText style={styles.authButtonText}>탈퇴하기</CustomText>
          </TouchableOpacity>
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
            <CustomText style={styles.modalTitle}>정말 탈퇴하시겠어요?</CustomText>

            <View style={styles.modalButtonWrapper}>
              <TouchableOpacity style={[styles.modalButton, styles.modalButtonPrimary]} onPress={() => navigation.replace('Mypage')}>
                <CustomText style={styles.modalButtonText}>아니요, 안 할래요</CustomText>
              </TouchableOpacity>

              <TouchableOpacity style={[styles.modalButton, styles.modalButtonSecondary]} onPress={() => handleDelete()}>
                <CustomText style={styles.modalButtonText}>네, 탈퇴 할래요</CustomText>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
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
    // position: 'relative',
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
    paddingHorizontal: 15,
    paddingTop: 10,
  },
  listText: {
    paddingBottom: 20,
    lineHeight: 24,
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
  buttonWrapper: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    width: '100%',
    padding: 10,
    borderTopLeftRadius: 10,
    borderTopRightRadius: 10,
    backgroundColor: '#FFFFFF',
  },
  authButton: {
    flex: 1,
    alignItems: 'center',
    padding: 10,
    borderRadius: 10,
  },
  authButtonPrimary: {
    backgroundColor: '#F2EAE3',
  },
  authButtonSecondary: {
    backgroundColor: '#D6CCC2',
  },
  authButtonText: {
    fontSize: 18,
    color: '#575553',
  },
});

export default AccountDeleteScreen;