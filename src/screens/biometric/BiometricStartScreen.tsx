import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import RNFS from 'react-native-fs';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';
import { biometricReader } from '../../services/biometricReader.ts';

const BiometricStartScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'BiometricStart'>;
  const navigation = useNavigation<Navigation>();

  useEffect(() => {
    async function readBiometricData() {
      console.log('파일 읽기 시작');

      const samsungHealthPath = `${RNFS.DownloadDirectoryPath}/Samsung Health`;

      const subFolders = await RNFS.readDir(samsungHealthPath);
      const targetFolder = subFolders.find(f => f.isDirectory());
      if (!targetFolder) {
        console.log('하위 폴더가 없습니다');
        return;
      }

      try {
        const results = await biometricReader(targetFolder.path);
        console.log('추출된 결과');
        console.log(results);
      } catch (error) {
        console.log('읽기 에러:', error);
      }
    }

    readBiometricData();

    const timer = setTimeout(() => {
      navigation.replace('BiometricComplete');
    }, 3000);

    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        <CustomText style={styles.title}>생체정보 입력을 시작할게요!</CustomText>
      </View>

      <View style={styles.imageContainer}>
        <FastImage source={require('../../assets/images/biometric_start.gif')} style={styles.image} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  textContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  title: {
    fontSize: 24,
  },
  imageContainer: {
    flex: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: 130,
    height: 130,
  },
});

export default BiometricStartScreen;