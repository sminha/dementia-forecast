import React from 'react';
import { View } from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';

type IconProps = {
  name: string;
  size: number;
  color?: string;
  style?: object;
};

const Icon: React.FC<IconProps> = ({ name, size, color = '#000000', style }) => {
  return (
    <View style={style}>
        <Ionicons name={name} size={size} color={color} />
    </View>
  );
};

export default Icon;