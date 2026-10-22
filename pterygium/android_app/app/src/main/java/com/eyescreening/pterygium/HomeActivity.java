package com.eyescreening.pterygium;

import android.content.Intent;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public final class HomeActivity extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        findViewById(R.id.openPterygiumButton).setOnClickListener(
                view -> startActivity(new Intent(this, MainActivity.class)));
        findViewById(R.id.openKeratoconusButton).setOnClickListener(
                view -> startActivity(new Intent(this, KeratoconusActivity.class)));
        findViewById(R.id.openDryEyeButton).setOnClickListener(
                view -> startActivity(new Intent(this, DryEyeActivity.class)));
    }
}
