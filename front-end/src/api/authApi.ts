import { loadTokens } from '../redux/actions/authAction.ts';

export const fetchDataWithToken = async () => {
  try {
    const { accessToken } = await loadTokens();

    if (!accessToken) {
      throw new Error('로그인된 상태가 아닙니다.');
    }

    const response = await fetch('BACKEND_URL', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
    });

    const data = response.json();
    return data;
  } catch (error) {
    console.error('API 요청 실패:', error);
    throw error;
  }
};