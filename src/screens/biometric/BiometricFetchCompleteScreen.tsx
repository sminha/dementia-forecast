import React, { useEffect, useState, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';

const BiometricCompleteScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'BiometricFetchComplete'>;
  const navigation = useNavigation<Navigation>();

  const [isAdditionalInput, setIsAdditionalInput] = useState<boolean>(false);
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const timer1 = setTimeout(() => {
      setIsAdditionalInput(true);
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 700,
        useNativeDriver: true,
      }).start();
    }, 3000);

    const timer2 = setTimeout(() => {
      navigation.replace('BiometricInput');
    }, 7000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [fadeAnim, navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        {!isAdditionalInput ? (
          <CustomText style={styles.title}>생체정보를 불러왔어요!</CustomText>)
        : (
          <Animated.View style={{ opacity: fadeAnim }}>
            <CustomText style={styles.title}>정확한 분석을 위해</CustomText>
            <CustomText style={styles.title}>신장과 체중은 직접 입력받고 있어요.</CustomText>
          </Animated.View>
        )}
      </View>

      <View style={styles.imageContainer}>
        <FastImage source={require('../../assets/images/scroll-down.gif')} style={styles.image} />
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

export default BiometricCompleteScreen;