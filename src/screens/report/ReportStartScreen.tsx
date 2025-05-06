import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';

const ReportStartScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'ReportStart'>;
  const navigation = useNavigation<Navigation>();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.replace('ReportResult');
    }, 3000);

    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        <CustomText style={styles.title}>알려주신 정보로{'\n'}치매 위험도를 진단하는 중이에요.{'\n\n'}잠시만 기다려주세요!</CustomText>
      </View>

      <View style={styles.imageContainer}>
        <FastImage source={require('../../assets/images/report_start.gif')} style={styles.image} />
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

export default ReportStartScreen;