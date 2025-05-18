import React, { useEffect } from 'react';
import { View, Image, StyleSheet, Linking } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { openExternalStoragePermission } from '../utils/permissionHelper.ts';
import RNFS from 'react-native-fs';

const SplashScreen = ({ navigation }: any) => {
  useEffect(() => {
    openExternalStoragePermission();

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

      const timer = setTimeout(() => {
        navigation.replace('Home');
      }, 3000);

      return () => clearTimeout(timer);
    };

    handleDeepLink();
  }, [navigation]);

  return (
    <View style={styles.container}>
      <Image source={require('../assets/images/logo.png')} style={styles.logo} />
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
});

export default SplashScreen;