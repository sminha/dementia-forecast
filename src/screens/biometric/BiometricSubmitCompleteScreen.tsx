import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { useSelector } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { RootState } from '../../redux/store.ts';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';

const BiometricSubmitCompleteScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'BiometricFetchComplete'>;
  const navigation = useNavigation<Navigation>();

  const biometric = useSelector((state: RootState) => state.biometric);
  const biometricInfo = useSelector((state: RootState) => state.biometric.biometricData);

  // console.log('biometric:', biometric);
  // { height: '180',
  //   weight: '60',
  //   biometricData: 
  //     [ { date: '2025-06-14',
  //         biometric_data_list: 
  //         [ { biometric_data_type: 'exercise',
  //             biometric_data_value: 
  //               [ { type: 1001,
  //                   start_time: '2025-05-19 10:10:31.482',
  //                   end_time: '2025-05-19 10:35:57.386',
  //                   duration: 1307411,
  //                   average_heart_rate: 111,
  // console.log('biometricInfo:', biometricInfo);
  // [ { date: '2025-06-14',
  //   biometric_data_list: 
  //     [ { biometric_data_type: 'exercise',
  //         biometric_data_value: 
  //         [ { type: 1001,
  //             start_time: '2025-05-19 10:10:31.482',
  //             end_time: '2025-05-19 10:35


  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.navigate('BiometricOverview', { from: 'BiometricSubmitComplete' });
    }, 3000);

    return () => clearTimeout(timer);
  }, [biometricInfo, navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        <CustomText style={styles.title}>생체정보 입력을 완료했어요!</CustomText>
      </View>

      <View style={styles.imageContainer}>
        <FastImage source={require('../../assets/images/complete.gif')} style={styles.image} />
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
    textAlign: 'center',
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

export default BiometricSubmitCompleteScreen;