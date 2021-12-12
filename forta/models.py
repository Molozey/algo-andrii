from django.db import models

class TickerIndicators(models.Model):
    ticker = models.OneToOneField('TickerInformation', models.DO_NOTHING, db_column='Ticker', primary_key=True)  # Field name made lowercase.
    update_date = models.DateTimeField(db_column='Update_Date', blank=True, null=True)  # Field name made lowercase.
    last_price = models.FloatField(db_column='Last_Price', blank=True, null=True)  # Field name made lowercase.
    market_cap = models.FloatField(db_column='Market_Cap', blank=True, null=True)  # Field name made lowercase.
    earnings = models.FloatField(db_column='Earnings', blank=True, null=True)  # Field name made lowercase.
    revenue_growth = models.FloatField(db_column='Revenue_Growth', blank=True, null=True)  # Field name made lowercase.
    price_to_earnings = models.FloatField(db_column='Price_To_Earnings', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'Ticker_Indicators'


class TickerInformation(models.Model):
    ticker = models.CharField(db_column='Ticker', primary_key=True, max_length=10)  # Field name made lowercase.
    name = models.CharField(db_column='Name', unique=True, max_length=256)  # Field name made lowercase.
    currency = models.CharField(db_column='Currency', max_length=45, blank=True, null=True)  # Field name made lowercase.
    country = models.CharField(db_column='Country', max_length=45, blank=True, null=True)  # Field name made lowercase.
    update_date = models.DateTimeField(db_column='Update_Date', blank=True, null=True)  # Field name made lowercase.
    last_price = models.FloatField(db_column='Last_Price', blank=True, null=True)  # Field name made lowercase.
    description = models.TextField(db_column='Description', blank=True, null=True)  # Field name made lowercase.
    logo_url = models.CharField(db_column='Logo_Url', max_length=100, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'Ticker_Information'


class TickerPrices(models.Model):
    ticker = models.OneToOneField(TickerInformation, models.DO_NOTHING, db_column='Ticker', primary_key=True)  # Field name made lowercase.
    update_date = models.DateTimeField(db_column='Update_Date', blank=True, null=True)  # Field name made lowercase.
    day_prices = models.JSONField(db_column='Day_Prices', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'Ticker_Prices'


class TickerSector(models.Model):
    ticker = models.OneToOneField(TickerInformation, models.DO_NOTHING, db_column='Ticker', primary_key=True)  # Field name made lowercase.
    sector = models.CharField(db_column='Sector', max_length=45, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'Ticker_Sector'


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'
