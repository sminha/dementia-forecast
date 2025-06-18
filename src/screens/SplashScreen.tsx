import React, { useEffect, useState } from 'react';
import { View, Image, StyleSheet, Linking, Text } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { openExternalStoragePermission } from '../utils/permissionHelper.ts';
import RNFS from 'react-native-fs';
import { loadTokens } from '../redux/actions/authAction.ts';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '../redux/store.ts';
import { fetchUser } from '../redux/slices/userSlice.ts';
import { fetchLifestyle } from '../redux/slices/lifestyleSlice.ts';

const SplashScreen = ({ navigation }: any) => {
  const dispatch = useDispatch<AppDispatch>();
  // const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const [showPermissionModal, setShowPermissionModal] = useState(true); // 모달 표시

  useEffect(() => {
    const timer1 = setTimeout(() => {
      setShowPermissionModal(false); // 모달 닫기
      openExternalStoragePermission(); // 권한 요청
    }, 3000);

    // openExternalStoragePermission();
    checkBiometricInfoExpiry();

    const handleDeepLink = async () => {
      const url = await Linking.getInitialURL();
      if (url) {
        const parsedUrl = new URL(url);
        const token = parsedUrl.searchParams.get('token');

        if (token) {
          AsyncStorage.setItem('accessToken', token);
          navigation.replace('Home');
          return;
        }
      }

      const samsungHealthPath = `${RNFS.DownloadDirectoryPath}/Samsung Health`;

      const readSamsungHealthFolder = async () => {
        const files = await RNFS.readDir(samsungHealthPath);
        console.log('Samsung Health 폴더 내 파일 목록:', files);
      };

      readSamsungHealthFolder();

      const timer2 = setTimeout(() => {
        navigation.replace('Home');
      }, 6000);

      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
      };
    };

    handleDeepLink();
  }, [navigation]);

  const checkBiometricInfoExpiry = async () => {
    try {
      const json = await AsyncStorage.getItem('biometricInfo');
      if (!json) return;

      const { savedAt } = JSON.parse(json);
      const now = Date.now();
      const twoWeeks = 14 * 24 * 60 * 60 * 1000;

      if (!savedAt || now - savedAt > twoWeeks) {
        await AsyncStorage.removeItem('biometricInfo');
        console.log('2주가 지나 생체 정보가 삭제되었습니다.');
      }
    } catch (error) {
      console.error('생체 정보 만료 체크 실패:', error);
    }
  };

  // 로그인 여부 확인
  useEffect(() => {
    const handleFetch = async () => {
      const { accessToken } = await loadTokens();
      if (!accessToken) {
        console.log('로그인 정보가 없습니다.');
        return;
      }

      dispatch(fetchUser(accessToken));
    };
    handleFetch();
  }, []);


  // 라이프스타일 입력 완료 여부 확인
  useEffect(() => {
    const handleFetchLifestyle = async () => {
      const { accessToken } = await loadTokens();
      // console.log('accessToken:', accessToken);
      if (!accessToken) {
        console.log('로그인 정보가 없습니다.');
        return;
      }

      dispatch(fetchLifestyle(accessToken));
    };

    handleFetchLifestyle();
  }, []);

  return (
    // <View style={styles.container}>
    //   <Image source={require('../assets/images/logo.png')} style={styles.logo} />
    // </View>
    <View style={styles.container}>
      <Image source={require('../assets/images/logo.png')} style={styles.logo} />
      {showPermissionModal && (
        <View style={styles.modal}>
          <Text style={styles.modalText}>
            앱 사용을 위해 외부 저장소 접근 권한이 필요합니다.{'\n'}치매예보의 모든 파일에 대한 접근을 허용해주세요.
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#EEE8E4',
  },
  logo: {
    width: 80,
    height: 80,
  },
  modal: {
    position: 'absolute',
    bottom: 50,
    backgroundColor: '#000a',
    padding: 20,
    borderRadius: 10,
  },
  modalText: {
    color: '#fff',
    textAlign: 'center',
  },
});

export default SplashScreen;