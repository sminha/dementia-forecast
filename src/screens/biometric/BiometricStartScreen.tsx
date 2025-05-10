import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';
// import { NativeModules } from 'react-native';

// const { HealthConnectModule } = NativeModules;

const BiometricStartScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'BiometricStart'>;
  const navigation = useNavigation<Navigation>();

  useEffect(() => {
    // fetchBiometricData();

    const timer = setTimeout(() => {
      navigation.replace('BiometricComplete');
    }, 3000);

    return () => clearTimeout(timer);
  }, [navigation]);

  // const fetchBiometricData = async () => {
  //   try {
  //     const heartRateData = await new Promise((resolve) => {
  //       HealthConnectModule.getHeartRateData((data: any) => {
  //         resolve(data);
  //       });
  //     });
  //     console.log('심박수 데이터:', heartRateData);

  //     const stepsData = await new Promise((resolve) => {
  //       HealthConnectModule.getStepsData((data: any) => {
  //         resolve(data);
  //       });
  //     });
  //     console.log('걸음 수 데이터:', stepsData);

  //     const sleepData = await new Promise((resolve) => {
  //       HealthConnectModule.getSleepData((data: any) => {
  //         resolve(data);
  //       });
  //     });
  //     console.log('수면 데이터:', sleepData);
  //   } catch (error) {
  //     console.error('생체정보 불러오기 오류:', error);
  //   }
  // };

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

// import React, { useEffect, useState } from 'react';
// import { View, Text, StyleSheet } from 'react-native';
// import { useNavigation } from '@react-navigation/native';
// import { StackNavigationProp } from '@react-navigation/stack';
// import { RootStackParamList } from '../../types/navigationTypes.ts';
// import FastImage from 'react-native-fast-image';
// // import { NativeModules } from 'react-native';

// // const { HealthConnectModule } = NativeModules;

// const BiometricStartScreen = () => {
//   type Navigation = StackNavigationProp<RootStackParamList, 'BiometricStart'>;
//   const navigation = useNavigation<Navigation>();

//   const [stepCount, setStepCount] = useState<number | null>(null); // 걸음수를 저장할 상태

//   useEffect(() => {
//     const fetchStepCount = async () => {
//       try {
//         // 삼성헬스에서 걸음수 데이터를 받아오는 부분
//         // 이 부분은 실제 삼성헬스 SDK에서 제공하는 API로 교체해야 함
//         const stepsData = await new Promise<number>((resolve) => {
//           // HealthConnectModule.getStepsData((data: number) => {
//           //   resolve(data); // 걸음 수 데이터
//           // });
//           // 임시로 1000 걸음으로 설정 (테스트용)
//           resolve(1000);
//         });
//         setStepCount(stepsData); // 받아온 데이터를 상태에 저장
//         console.log('걸음 수 데이터:', stepsData); // 콘솔에 찍어서 확인
//       } catch (error) {
//         console.error('걸음 수 데이터를 가져오는 데 실패했습니다:', error);
//       }
//     };

//     fetchStepCount(); // 컴포넌트가 마운트되면 데이터 요청

//     const timer = setTimeout(() => {
//       navigation.replace('BiometricComplete');
//     }, 3000);

//     return () => clearTimeout(timer);
//   }, [navigation]);

//   return (
//     <View style={styles.container}>
//       <View style={styles.textContainer}>
//         <Text style={styles.title}>생체정보 입력을 시작할게요!</Text>
//         {stepCount !== null ? (
//           <Text style={styles.stepCount}>걸음 수: {stepCount}</Text>
//         ) : (
//           <Text>걸음 수를 가져오는 중...</Text>
//         )}
//       </View>

//       <View style={styles.imageContainer}>
//         <FastImage source={require('../../assets/images/biometric_start.gif')} style={styles.image} />
//       </View>
//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: '#FFFFFF',
//     justifyContent: 'center',
//     alignItems: 'center',
//   },
//   textContainer: {
//     alignItems: 'center',
//     justifyContent: 'center',
//     marginBottom: 20,
//   },
//   title: {
//     fontSize: 24,
//     marginBottom: 10,
//   },
//   stepCount: {
//     fontSize: 18,
//     marginTop: 10,
//   },
//   imageContainer: {
//     alignItems: 'center',
//     justifyContent: 'center',
//   },
//   image: {
//     width: 130,
//     height: 130,
//   },
// });

// export default BiometricStartScreen;
