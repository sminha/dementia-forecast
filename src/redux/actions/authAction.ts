import AsyncStorage from '@react-native-async-storage/async-storage';

export const storeTokens = async (accessToken: string, refreshToken: string) => {
  try {
    await AsyncStorage.setItem('accessToken', accessToken);
    await AsyncStorage.setItem('refreshToken', refreshToken);
  } catch (error) {
    console.error('토큰 저장 실패:', error);
  }
};

export const loadTokens = async () => {
  try {
    const accessToken = await AsyncStorage.getItem('accessToken');
    const refreshToken = await AsyncStorage.getItem('refreshToken');

    return { accessToken, refreshToken };
  } catch (error) {
    console.error('토큰 불러오기 실패:', error);
    return { accessToken: null, refreshToken: null };
  }
};