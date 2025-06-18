import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';

const LifestyleCompleteScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'LifestyleComplete'>;
  const navigation = useNavigation<Navigation>();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.navigate('Home');
    }, 3000);

    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        <CustomText style={styles.title}>라이프스타일 입력을 완료했어요!</CustomText>
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

export default LifestyleCompleteScreen;