import React from 'react';
import { View, TouchableOpacity, Image, Text, ScrollView, StyleSheet } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const forceBreakText = (text: string) => {
  return text.split('').join('\u200B'); // 문자 사이마다 zero-width space 삽입
};

const DementiaInfoDetailScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'EmailLogin'>;
  const navigation = useNavigation<Navigation>();

  const route = useRoute();
  const { item } = route.params;

  return (
    <View style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.imageContainer}>
          <Image
            source={require('../../assets/images/dementia_info_detail.png')}
            style={styles.background}
            resizeMode="cover"
          />

          <Image
            source={item.image}
            style={styles.overlayImage}
            resizeMode="cover"
          />

          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <Icon name="chevron-back" size={16} color="#FFFFFF" />
          </TouchableOpacity>

          <View style={styles.titleWrapper}>
            <Text style={styles.title}>{item.title}</Text>
          </View>
        </View>

        <View style={styles.contentWrapper}>
          {item.content.split('\n\n').map((paragraph, index) => (
            <CustomText key={index} style={styles.content}>
              {'  '}{forceBreakText(paragraph)}
            </CustomText>
          ))}
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  imageContainer: {
    position: 'relative',
    width: '100%',
    height: 260,
    justifyContent: 'center',
    alignItems: 'center',
  },
  background: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  overlayImage: {
    position: 'absolute',
    width: 180,
    height: 180,
    opacity: 0.2,
    zIndex: 1,
  },
  backButton: {
    position: 'absolute',
    top: 30,
    left: 26,
    zIndex: 2,
  },
  titleWrapper: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 22,
    color: '#FFFFFF',
    fontWeight: '600',
    zIndex: 3,
  },
  contentWrapper: {
    paddingHorizontal: 26,
    paddingVertical: 26,
  },
  content: {
    marginBottom: 16,
    textAlign: 'justify',
    lineHeight: 28,
    fontSize: 20,
    color: '#626262',
  },
});


export default DementiaInfoDetailScreen;