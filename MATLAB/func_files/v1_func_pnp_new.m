function obj = v1_func_pnp_new(in1)
q0 = in1(1,:);
q1 = in1(2,:);
q2 = in1(3,:);
q3 = in1(4,:);
t98 = q0.^2;
t99 = q1.^2;
t100 = q2.^2;
t101 = q3.^2;
obj = [t98.^2;q0.*q1.*t98;q0.*q2.*t98;q0.*q3.*t98;t98.*t99;q1.*q2.*t98;q1.*q3.*t98;t100.*t98;q2.*q3.*t98;t101.*t98;t98;q0.*q1.*t99;q0.*q2.*t99;q0.*q3.*t99;q0.*q1.*t100;q0.*q1.*q2.*q3;q0.*q1.*t101;q0.*q1;q0.*q2.*t100;q0.*q3.*t100;q0.*q2.*t101;q0.*q2;q0.*q3.*t101;q0.*q3;t99.^2;q1.*q2.*t99;q1.*q3.*t99;t100.*t99;q2.*q3.*t99;t101.*t99;t99;q1.*q2.*t100;q1.*q3.*t100;q1.*q2.*t101;q1.*q2;q1.*q3.*t101;q1.*q3];
